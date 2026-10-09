"""Tk review interface and table helpers for the human QC gate.

The review stage deliberately changes only the two compact, canonical TSV
files.  It never writes a second FIF, copies PREP reports, or emits figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd


ANNOTATION_COLUMNS = (
    "onset",
    "duration",
    "trial_type",
    "annotation_source",
    "review_status",
    "reviewed_description",
    "reviewer",
    "reviewed_at",
    "review_notes",
)
CHANNEL_REVIEW_COLUMNS = (
    "manual_decision",
    "final_decision",
    "decision_reason",
    "reviewer",
    "reviewed_at",
    "review_notes",
)
ANNOTATION_DECISIONS = ("unreviewed", "keep", "approved_bad", "dismissed")
CHANNEL_DECISIONS = ("unreviewed", "retain", "interpolate", "exclude_from_ica")


@dataclass(frozen=True)
class ReviewRecord:
    """The three input/output files reviewed together for a single run."""

    label: str
    raw_path: Path
    annotations_path: Path
    channels_path: Path


def _string_column(table: pd.DataFrame, column: str, default: str = "") -> None:
    if column not in table:
        table[column] = default
    table[column] = table[column].fillna(default).astype(str)


def normalise_annotation_table(table: pd.DataFrame) -> pd.DataFrame:
    """Add review columns to an old candidate table without losing its content."""
    table = table.copy()
    required = {"onset", "duration", "trial_type", "annotation_source", "review_status"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Annotation table is missing required columns: {sorted(missing)}")
    for column in ("onset", "duration"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    for column in ANNOTATION_COLUMNS[2:]:
        _string_column(table, column, "unreviewed" if column == "review_status" else "")
    return table.loc[:, list(ANNOTATION_COLUMNS)]


def normalise_channel_table(table: pd.DataFrame) -> pd.DataFrame:
    """Add review columns to an old PREP channel table without losing evidence."""
    table = table.copy()
    required = {"channel", "channel_type", "automatic_status", "automatic_methods"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Channel QC table is missing required columns: {sorted(missing)}")
    for column in CHANNEL_REVIEW_COLUMNS:
        default = "unreviewed" if column in {"manual_decision", "final_decision"} else ""
        _string_column(table, column, default)
    return table


def read_review_tables(record: ReviewRecord) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a run's two canonical review tables, normalising old schemas."""
    annotations = normalise_annotation_table(pd.read_csv(record.annotations_path, sep="\t"))
    channels = normalise_channel_table(pd.read_csv(record.channels_path, sep="\t"))
    return annotations, channels


def write_review_tables(
    record: ReviewRecord,
    annotations: pd.DataFrame,
    channels: pd.DataFrame,
) -> None:
    """Atomically replace only the two review tables for the selected run."""
    annotations = normalise_annotation_table(annotations)
    channels = normalise_channel_table(channels)
    annotation_tmp = record.annotations_path.with_suffix(".tsv.tmp")
    channel_tmp = record.channels_path.with_suffix(".tsv.tmp")
    annotations.to_csv(annotation_tmp, sep="\t", index=False)
    channels.to_csv(channel_tmp, sep="\t", index=False)
    annotation_tmp.replace(record.annotations_path)
    channel_tmp.replace(record.channels_path)


def approved_bad_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return only reviewer-approved spans to turn into MNE ``BAD_*`` labels."""
    annotations = normalise_annotation_table(annotations)
    approved = annotations.loc[annotations["review_status"] == "approved_bad"].copy()
    approved = approved.loc[approved["reviewed_description"].str.startswith("BAD_")]
    return approved.loc[:, ["onset", "duration", "reviewed_description"]]


def discover_review_records(
    derivatives_path: Path,
    condition: str,
    subjects: Iterable[int],
    runs: Iterable[int],
) -> list[ReviewRecord]:
    """Find complete Stage 1/2 checkpoints in deterministic review order."""
    records: list[ReviewRecord] = []
    for subject_number in subjects:
        subject = f"sub-{int(subject_number):02d}"
        directory = derivatives_path / subject / "ses-01" / "eeg"
        for run_number in runs:
            base = f"{subject}_ses-01_task-{condition}_run-{int(run_number):02d}"
            record = ReviewRecord(
                label=base,
                raw_path=directory / f"{base}_desc-minproc_raw.fif",
                annotations_path=directory / f"{base}_desc-annotationsqc.tsv",
                channels_path=directory / f"{base}_desc-channelqc.tsv",
            )
            missing = [path.name for path in (record.raw_path, record.annotations_path, record.channels_path) if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Incomplete QC checkpoint for {base}: missing {', '.join(missing)}")
            records.append(record)
    return records


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _bad_label(trial_type: str) -> str:
    labels = {
        "MUSCLE_CANDIDATE": "BAD_MUSCLE",
        "AMPLITUDE_CANDIDATE": "BAD_AMPLITUDE",
    }
    return labels.get(trial_type, "BAD_MANUAL")


class QCReviewApplication:
    """A small local GUI for run-wise annotation and channel decisions."""

    def __init__(self, root, records: list[ReviewRecord], reviewer: str = ""):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk, self.ttk = tk, ttk
        self.records = records
        self.reviewer = tk.StringVar(value=reviewer)
        self.record_name = tk.StringVar(value=records[0].label)
        self.start_seconds = tk.DoubleVar(value=0.0)
        self.window_seconds = tk.DoubleVar(value=15.0)
        self.annotation_onset = tk.DoubleVar(value=0.0)
        self.annotation_duration = tk.DoubleVar(value=1.0)
        self.annotation_bad_label = tk.StringVar(value="BAD_MANUAL")
        self.review_notes = tk.StringVar(value="")
        self.view_highpass = tk.StringVar(value="")
        self.view_lowpass = tk.StringVar(value="")
        self.view_notch = tk.StringVar(value="50")
        self.view_average_reference = tk.BooleanVar(value=False)
        self.trace_amplitude_uv = tk.StringVar(value="100")
        self.status = tk.StringVar(value="")
        self.annotation_selection_status = tk.StringVar(value="No artifacts selected.")
        self.channel_details = tk.StringVar(value="Select a channel to see its PREP evidence.")
        self.current_record: ReviewRecord | None = None
        self.raw = None
        self.annotations = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        self.channels = pd.DataFrame()
        self.annotation_sort_column = "onset"
        self.annotation_sort_ascending = True
        self._display_cache_key = None
        self._display_cache = None
        self.visible_channel_indices: set[int] = set()
        self.selected_time_seconds: float | None = None

        root.title("Spell-IT-OUT QC review")
        root.geometry("1500x900")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        controls = ttk.Frame(root, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(9, weight=1)
        ttk.Label(controls, text="Run").grid(row=0, column=0, sticky="w")
        self.record_combo = ttk.Combobox(controls, textvariable=self.record_name, values=[item.label for item in records], state="readonly", width=52)
        self.record_combo.grid(row=0, column=1, padx=(4, 12), sticky="w")
        self.record_combo.bind("<<ComboboxSelected>>", self.load_selected_record)
        ttk.Label(controls, text="Reviewer").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.reviewer, width=18).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(controls, text="Start (s)").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.start_seconds, width=8).grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="Window (s)").grid(row=0, column=6, sticky="w")
        ttk.Spinbox(controls, from_=2, to=120, increment=1, textvariable=self.window_seconds, width=6, command=self.draw_trace).grid(row=0, column=7, padx=4)
        ttk.Label(controls, text="Trace limit ±µV").grid(row=0, column=8, padx=(12, 0), sticky="w")
        amplitude_spinbox = ttk.Spinbox(
            controls,
            from_=10,
            to=2000,
            increment=10,
            textvariable=self.trace_amplitude_uv,
            width=6,
            command=self.draw_trace,
        )
        amplitude_spinbox.grid(row=0, column=9, padx=4)
        amplitude_spinbox.bind("<Return>", lambda _event: self.draw_trace())
        amplitude_spinbox.bind("<FocusOut>", lambda _event: self.draw_trace())
        ttk.Button(controls, text="◀", command=lambda: self.shift_window(-1)).grid(row=0, column=10, padx=(8, 2))
        ttk.Button(controls, text="▶", command=lambda: self.shift_window(1)).grid(row=0, column=11, sticky="w")
        ttk.Button(controls, text="Save now", command=self.save_current).grid(row=0, column=12, padx=(12, 0))
        view_controls = ttk.LabelFrame(controls, text="Display preprocessing only — never saved or used by PREP", padding=5)
        view_controls.grid(row=1, column=0, columnspan=11, pady=(8, 0), sticky="ew")
        ttk.Label(view_controls, text="High-pass (Hz)").grid(row=0, column=0, sticky="w")
        ttk.Entry(view_controls, textvariable=self.view_highpass, width=7).grid(row=0, column=1, padx=(3, 9))
        ttk.Label(view_controls, text="Low-pass (Hz)").grid(row=0, column=2, sticky="w")
        ttk.Entry(view_controls, textvariable=self.view_lowpass, width=7).grid(row=0, column=3, padx=(3, 9))
        ttk.Label(view_controls, text="Notch (Hz; comma-separated)").grid(row=0, column=4, sticky="w")
        ttk.Entry(view_controls, textvariable=self.view_notch, width=14).grid(row=0, column=5, padx=(3, 9))
        ttk.Checkbutton(view_controls, text="Average reference", variable=self.view_average_reference).grid(row=0, column=6, padx=(0, 9))
        ttk.Button(view_controls, text="Apply to view", command=self.apply_display_preprocessing).grid(row=0, column=7, padx=(0, 4))
        ttk.Button(view_controls, text="Reset view", command=self.reset_display_preprocessing).grid(row=0, column=8)

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        trace_frame = ttk.Labelframe(main, text="Selected-channel trace (µV)", padding=6)
        right = ttk.Frame(main)
        main.add(trace_frame, weight=3)
        main.add(right, weight=2)
        trace_frame.columnconfigure(0, weight=1)
        trace_frame.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.axes = [self.figure.add_subplot(111)]
        self.canvas = FigureCanvasTkAgg(self.figure, master=trace_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self.select_trace_time)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.annotation_tab = ttk.Frame(notebook, padding=6)
        self.channel_tab = ttk.Frame(notebook, padding=6)
        notebook.add(self.annotation_tab, text="Artifact annotations")
        notebook.add(self.channel_tab, text="Channel decisions")
        self._build_annotation_tab()
        self._build_channel_tab()
        ttk.Label(root, textvariable=self.status, anchor="w", padding=(8, 0, 8, 8)).grid(row=2, column=0, sticky="ew")
        root.bind("<Control-s>", lambda event: self.save_current())
        self.bind_global_shortcuts(root)
        self.load_record(records[0])

    def bind_global_shortcuts(self, root) -> None:
        """Bind explicit keys before every widget's normal key handling."""
        actions = {
            "<KeyPress-u>": self.update_annotation_span,
            "<KeyPress-U>": self.update_annotation_span,
            "<KeyPress-k>": lambda: self.set_annotation_decision("keep"),
            "<KeyPress-K>": lambda: self.set_annotation_decision("keep"),
            "<KeyPress-a>": lambda: self.set_annotation_decision("approved_bad"),
            "<KeyPress-A>": lambda: self.set_annotation_decision("approved_bad"),
            "<KeyPress-d>": lambda: self.set_annotation_decision("dismissed"),
            "<KeyPress-D>": lambda: self.set_annotation_decision("dismissed"),
            "<KeyPress-plus>": self.add_bad_annotation,
            "<KeyPress-KP_Add>": self.add_bad_annotation,
            "<Left>": lambda: self.shift_window(-1),
            "<Right>": lambda: self.shift_window(1),
        }
        shortcut_tag = "SpellItOutShortcuts"
        for sequence, action in actions.items():
            handler = lambda event, callback=action: self.run_shortcut(event, callback)
            root.bind_all(sequence, handler, add="+")
            root.bind_class(shortcut_tag, sequence, handler, add="+")

        def add_shortcut_tag(widget) -> None:
            tags = widget.bindtags()
            if shortcut_tag not in tags:
                # This custom tag is first, so its callback is reached even
                # when the focused widget would otherwise consume the key.
                widget.bindtags((shortcut_tag, *tags))
            for child in widget.winfo_children():
                add_shortcut_tag(child)

        add_shortcut_tag(root)

    def run_shortcut(self, event, action):
        """Execute a shortcut unless the user is editing a text control."""
        widget_class = event.widget.winfo_class()
        if widget_class in {
            "Entry",
            "Spinbox",
            "Text",
            "TEntry",
            "TSpinbox",
            "TCombobox",
        }:
            return None
        # On Windows, bit 0x8 is not reliable for detecting Alt and can be set
        # by keyboard/Num-Lock state.  Treating it as a blocker made every
        # shortcut silently return on affected keyboards.
        if event.state & 0x4:
            return None
        action()
        return "break"

    def _build_annotation_tab(self) -> None:
        ttk = self.ttk
        self.annotation_tab.columnconfigure(0, weight=1)
        self.annotation_tab.rowconfigure(0, weight=1)
        columns = ("onset", "duration", "trial_type", "review_status", "reviewed_description")
        self.annotation_tree = ttk.Treeview(
            self.annotation_tab,
            columns=columns,
            show="headings",
            height=18,
            selectmode="extended",
        )
        widths = (72, 75, 145, 115, 140)
        for column, width in zip(columns, widths):
            self.annotation_tree.heading(
                column,
                text=column.replace("_", " "),
                command=lambda value=column: self.sort_annotations(value),
            )
            self.annotation_tree.column(column, width=width, anchor="w")
        self.annotation_tree.grid(row=0, column=0, sticky="nsew")
        self.annotation_tree.bind("<<TreeviewSelect>>", self.select_annotation)
        edit = ttk.Frame(self.annotation_tab)
        edit.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(edit, text="Onset").grid(row=0, column=0)
        ttk.Entry(edit, textvariable=self.annotation_onset, width=8).grid(row=0, column=1, padx=3)
        ttk.Label(edit, text="Duration").grid(row=0, column=2)
        ttk.Entry(edit, textvariable=self.annotation_duration, width=8).grid(row=0, column=3, padx=3)
        ttk.Label(edit, text="BAD label").grid(row=0, column=4)
        ttk.Combobox(edit, textvariable=self.annotation_bad_label, values=("BAD_MUSCLE", "BAD_AMPLITUDE", "BAD_MOVEMENT", "BAD_MANUAL"), state="readonly", width=15).grid(row=0, column=5, padx=3)
        ttk.Button(edit, text="Update span (U)", command=self.update_annotation_span).grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(edit, text="Keep (K)", command=lambda: self.set_annotation_decision("keep")).grid(row=1, column=2, pady=5, sticky="ew")
        ttk.Button(edit, text="Approve BAD (A)", command=lambda: self.set_annotation_decision("approved_bad")).grid(row=1, column=3, padx=3, pady=5, sticky="ew")
        ttk.Button(edit, text="Dismiss (D)", command=lambda: self.set_annotation_decision("dismissed")).grid(row=1, column=4, pady=5, sticky="ew")
        ttk.Button(edit, text="Add BAD span (+)", command=self.add_bad_annotation).grid(row=1, column=5, padx=3, pady=5, sticky="ew")
        ttk.Button(edit, text="Add BLINK", command=self.add_blink_annotation).grid(row=1, column=6, padx=3, pady=5, sticky="ew")
        ttk.Label(edit, text="Notes").grid(row=2, column=0, pady=(3, 0), sticky="w")
        ttk.Entry(edit, textvariable=self.review_notes, width=52).grid(row=2, column=1, columnspan=5, pady=(3, 0), sticky="ew")
        ttk.Label(edit, textvariable=self.annotation_selection_status, anchor="w").grid(
            row=3, column=0, columnspan=6, pady=(5, 0), sticky="ew"
        )

    def _build_channel_tab(self) -> None:
        ttk = self.ttk
        self.channel_tab.columnconfigure(0, weight=1)
        self.channel_tab.rowconfigure(0, weight=1)
        columns = ("show", "channel", "automatic_status", "automatic_methods", "final_decision")
        self.channel_tree = ttk.Treeview(
            self.channel_tab,
            columns=columns,
            show="headings",
            height=19,
            selectmode="extended",
        )
        for column, width in zip(columns, (48, 80, 105, 155, 130)):
            self.channel_tree.heading(column, text=column.replace("_", " "))
            self.channel_tree.column(column, width=width, anchor="center" if column == "show" else "w")
        self.channel_tree.grid(row=0, column=0, sticky="nsew")
        self.channel_tree.bind("<<TreeviewSelect>>", self.select_channel)
        self.channel_tree.bind("<Button-1>", self.toggle_channel_visibility)
        decisions = ttk.Frame(self.channel_tab)
        decisions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for column, (label, decision) in enumerate((("Retain", "retain"), ("Interpolate", "interpolate"), ("Exclude ICA", "exclude_from_ica"), ("Clear decision", "unreviewed"))):
            ttk.Button(decisions, text=label, command=lambda value=decision: self.set_channel_decision(value)).grid(row=0, column=column, padx=2, sticky="ew")
            decisions.columnconfigure(column, weight=1)
        ttk.Button(decisions, text="Show all EEG", command=self.show_all_eeg_channels).grid(row=1, column=0, columnspan=2, pady=(7, 0), sticky="ew")
        ttk.Button(decisions, text="Hide all", command=self.hide_all_channels).grid(row=1, column=2, columnspan=2, padx=(4, 0), pady=(7, 0), sticky="ew")
        ttk.Label(decisions, text="Notes").grid(row=2, column=0, pady=(7, 0), sticky="w")
        ttk.Entry(decisions, textvariable=self.review_notes, width=45).grid(row=2, column=1, columnspan=3, pady=(7, 0), sticky="ew")
        ttk.Label(self.channel_tab, textvariable=self.channel_details, justify="left", wraplength=510).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def load_selected_record(self, _event=None) -> None:
        record = next(item for item in self.records if item.label == self.record_name.get())
        self.save_current()
        self.load_record(record)

    def load_record(self, record: ReviewRecord) -> None:
        is_initial_load = self.current_record is None
        previously_visible = (
            set(self.channels.loc[list(self.visible_channel_indices), "channel"])
            if not self.channels.empty and self.visible_channel_indices
            else set()
        )
        self.current_record = record
        self.raw = mne.io.read_raw_fif(record.raw_path, preload=False, verbose="error")
        self.annotations, self.channels = read_review_tables(record)
        self.record_name.set(record.label)
        self.start_seconds.set(0.0)
        self._display_cache_key = None
        self._display_cache = None
        self.selected_time_seconds = None
        self.visible_channel_indices = set(
            self.channels.index[self.channels["channel"].isin(previously_visible)]
        )
        if is_initial_load and not self.channels.empty:
            self.visible_channel_indices = {self.channels.index[0]}
        self.refresh_tables()
        self.status.set(f"Loaded {record.label}. Changes autosave after each decision.")

    def refresh_tables(
        self,
        annotation_selection: list[int] | None = None,
        channel_selection: list[int] | None = None,
    ) -> None:
        if annotation_selection is None:
            annotation_selection = self._selected_indices(self.annotation_tree, "a")
        if channel_selection is None:
            channel_selection = self._selected_indices(self.channel_tree, "c")
        for tree in (self.annotation_tree, self.channel_tree):
            tree.delete(*tree.get_children())
        sorted_annotations = self.annotations.sort_values(
            self.annotation_sort_column,
            ascending=self.annotation_sort_ascending,
            kind="stable",
        )
        direction = "▲" if self.annotation_sort_ascending else "▼"
        for column in self.annotation_tree["columns"]:
            label = column.replace("_", " ")
            if column == self.annotation_sort_column:
                label = f"{label} {direction}"
            self.annotation_tree.heading(column, text=label)
        for index, row in sorted_annotations.iterrows():
            self.annotation_tree.insert("", "end", iid=f"a{index}", values=(f"{row.onset:.3f}", f"{row.duration:.3f}", row.trial_type, row.review_status, row.reviewed_description))
        for index, row in self.channels.iterrows():
            visible = "☑" if index in self.visible_channel_indices else "☐"
            self.channel_tree.insert("", "end", iid=f"c{index}", values=(visible, row.channel, row.automatic_status, row.automatic_methods, row.final_decision))
        valid_channel_selection = [
            f"c{index}" for index in channel_selection if index in self.channels.index
        ]
        if valid_channel_selection:
            self.channel_tree.selection_set(valid_channel_selection)
        elif not self.channels.empty:
            self.channel_tree.selection_set(f"c{self.channels.index[0]}")
        valid_annotation_selection = [
            f"a{index}" for index in annotation_selection if index in self.annotations.index
        ]
        if valid_annotation_selection:
            self.annotation_tree.selection_set(valid_annotation_selection)
            self.annotation_tree.focus(valid_annotation_selection[0])
            self.annotation_tree.see(valid_annotation_selection[0])
            self.select_annotation()
        else:
            self.update_annotation_selection_status()
        if not self.channels.empty:
            self.select_channel()

    def sort_annotations(self, column: str) -> None:
        selected = self._selected_indices(self.annotation_tree, "a")
        if column == self.annotation_sort_column:
            self.annotation_sort_ascending = not self.annotation_sort_ascending
        else:
            self.annotation_sort_column = column
            self.annotation_sort_ascending = True
        self.refresh_tables(annotation_selection=selected)

    def _selected_index(self, tree, prefix: str) -> int | None:
        selected = tree.selection()
        return int(selected[0][len(prefix):]) if selected else None

    def _selected_indices(self, tree, prefix: str) -> list[int]:
        return [int(item[len(prefix):]) for item in tree.selection()]

    def toggle_channel_visibility(self, event):
        """Toggle only the checkbox column; row selection remains for decisions."""
        if self.channel_tree.identify_region(event.x, event.y) != "cell":
            return None
        if self.channel_tree.identify_column(event.x) != "#1":
            return None
        item = self.channel_tree.identify_row(event.y)
        if not item:
            return "break"
        index = int(item[1:])
        if index in self.visible_channel_indices:
            self.visible_channel_indices.remove(index)
        else:
            self.visible_channel_indices.add(index)
        self.refresh_tables()
        return "break"

    def select_annotation(self, _event=None) -> None:
        self.update_annotation_selection_status()
        index = self._selected_index(self.annotation_tree, "a")
        if index is None:
            return
        row = self.annotations.loc[index]
        self.annotation_onset.set(float(row.onset))
        self.annotation_duration.set(float(row.duration))
        if str(row.reviewed_description).startswith("BAD_"):
            self.annotation_bad_label.set(row.reviewed_description)
        self.review_notes.set(str(row.review_notes))
        self.start_seconds.set(max(0.0, float(row.onset) - self.window_seconds.get() / 3))
        self.draw_trace()

    def update_annotation_selection_status(self) -> None:
        count = len(self._selected_indices(self.annotation_tree, "a"))
        self.annotation_selection_status.set(
            f"{count} artifact{'s' if count != 1 else ''} selected."
        )

    def select_trace_time(self, event) -> None:
        """Mark the time clicked in any displayed EEG trace."""
        if event.button != 1 or event.inaxes not in self.axes or event.xdata is None:
            return
        self.selected_time_seconds = float(event.xdata)
        self.draw_trace()
        self.annotation_onset.set(self.selected_time_seconds)

    def select_channel(self, _event=None) -> None:
        index = self._selected_index(self.channel_tree, "c")
        if index is None:
            return
        row = self.channels.loc[index]
        self.review_notes.set(str(row.review_notes))
        self.channel_details.set(
            f"{row.channel}: {row.automatic_status}; methods: {row.automatic_methods or 'none'}\n"
            f"Correlation bad-window fraction: {row.get('correlation_bad_window_fraction', np.nan):.4g}; "
            f"median max correlation: {row.get('correlation_median_maximum', np.nan):.4g}\n"
            f"PSD z (low/mid/high): {row.get('psd_low_band_zscore', np.nan):.3g} / "
            f"{row.get('psd_mid_band_zscore', np.nan):.3g} / {row.get('psd_high_band_zscore', np.nan):.3g}"
        )
        self.draw_trace()

    def show_all_eeg_channels(self) -> None:
        eeg_indices = self.channels.index[
            self.channels["channel_type"].str.upper() == "EEG"
        ].tolist()
        self.visible_channel_indices.update(eeg_indices)
        self.refresh_tables()

    def hide_all_channels(self) -> None:
        self.visible_channel_indices.clear()
        self.refresh_tables()

    def _trace_limit_uv(self) -> float:
        try:
            limit = float(self.trace_amplitude_uv.get())
        except ValueError as error:
            raise ValueError("Trace limit must be a positive number in µV.") from error
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("Trace limit must be a positive number in µV.")
        return limit

    def draw_trace(self, *_args) -> None:
        if self.raw is None or self.channels.empty:
            return
        channel_indices = sorted(self.visible_channel_indices)
        if not channel_indices:
            self.figure.clear()
            self.canvas.draw_idle()
            self.status.set("Enable one or more channel checkboxes to display traces.")
            return
        try:
            trace_limit = self._trace_limit_uv()
        except ValueError as error:
            self.status.set(str(error))
            return
        channels = self.channels.loc[channel_indices, "channel"].tolist()
        duration = float(self.raw.times[-1])
        window = max(0.1, float(self.window_seconds.get()))
        start = min(max(0.0, float(self.start_seconds.get())), max(0.0, duration - window))
        self.start_seconds.set(start)
        stop = min(duration, start + window)
        try:
            data, times = self._display_data(channels, start, stop)
        except (RuntimeError, ValueError) as error:
            self.status.set(f"Display preprocessing was not applied: {error}")
            start_sample = int(start * self.raw.info["sfreq"])
            stop_sample = int(stop * self.raw.info["sfreq"])
            data = self.raw.get_data(picks=channels, start=start_sample, stop=stop_sample) * 1e6
            times = np.arange(data.shape[1]) / self.raw.info["sfreq"] + start
        self.figure.clear()
        axes = self.figure.subplots(len(channels), 1, sharex=True, squeeze=False).ravel().tolist()
        selected_annotation_indices = set(self._selected_indices(self.annotation_tree, "a"))
        colours = {"unreviewed": "#f0ad4e", "keep": "#5cb85c", "approved_bad": "#d9534f", "dismissed": "#bdbdbd"}
        for channel, channel_data, axis in zip(channels, data, axes):
            axis.plot(times, channel_data, color="#1f4e79", linewidth=0.7)
            for annotation_index, row in self.annotations.iterrows():
                onset, end = float(row.onset), float(row.onset + row.duration)
                if end < start or onset > stop:
                    continue
                if annotation_index in selected_annotation_indices:
                    axis.axvspan(onset, end, facecolor="none", edgecolor="#111111", linewidth=2.2, zorder=4)
                    axis.axvline(onset, color="#111111", linewidth=1.5, linestyle="--", zorder=5)
                    axis.axvline(end, color="#111111", linewidth=1.5, linestyle="--", zorder=5)
                else:
                    axis.axvspan(onset, end, color=colours.get(row.review_status, "#f0ad4e"), alpha=0.22)
            axis.set_ylabel(f"{channel}\nµV", rotation=0, ha="right", va="center")
            axis.set_ylim(-trace_limit, trace_limit)
            axis.grid(alpha=0.25)
        if selected_annotation_indices:
            selected = self.annotations.loc[next(iter(selected_annotation_indices))]
            axes[0].annotate(
                f"Selected: {len(selected_annotation_indices)} event(s)" if len(selected_annotation_indices) > 1 else f"Selected: {selected.trial_type}",
                xy=(float(selected.onset), 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -4),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#111111", "alpha": 0.9},
                zorder=6,
            )
        if self.selected_time_seconds is not None and start <= self.selected_time_seconds <= stop:
            for axis in axes:
                axis.axvline(
                    self.selected_time_seconds,
                    color="#7b1fa2",
                    linewidth=1.4,
                    linestyle="-",
                    zorder=7,
                )
            axes[0].annotate(
                f"t = {self.selected_time_seconds:.3f} s",
                xy=(self.selected_time_seconds, 0),
                xycoords=("data", "axes fraction"),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
                color="#4a148c",
                bbox={"facecolor": "white", "edgecolor": "#7b1fa2", "alpha": 0.9},
                zorder=8,
            )
        axes[-1].set_xlabel("Time (s)")
        for axis in axes:
            axis.set_xlim(start, stop)
        self.axes = axes
        self.canvas.draw_idle()

    def shift_window(self, direction: int) -> None:
        self.start_seconds.set(max(0.0, self.start_seconds.get() + direction * self.window_seconds.get() * 0.8))
        self.draw_trace()

    def apply_display_preprocessing(self) -> None:
        """Refresh the plot using filters on an in-memory copy of its time window."""
        self._display_cache_key = None
        self._display_cache = None
        self.draw_trace()

    def reset_display_preprocessing(self) -> None:
        self.view_highpass.set("")
        self.view_lowpass.set("")
        self.view_notch.set("")
        self.view_average_reference.set(False)
        self.apply_display_preprocessing()

    @staticmethod
    def _optional_frequency(value: str, label: str) -> float | None:
        value = value.strip()
        if not value:
            return None
        frequency = float(value)
        if frequency <= 0:
            raise ValueError(f"{label} must be a positive frequency.")
        return frequency

    def _display_settings(self) -> tuple[float | None, float | None, tuple[float, ...], bool]:
        highpass = self._optional_frequency(self.view_highpass.get(), "High-pass")
        lowpass = self._optional_frequency(self.view_lowpass.get(), "Low-pass")
        if highpass is not None and lowpass is not None and highpass >= lowpass:
            raise ValueError("High-pass must be lower than low-pass.")
        notch = tuple(
            float(value.strip())
            for value in self.view_notch.get().split(",")
            if value.strip()
        )
        if any(frequency <= 0 for frequency in notch):
            raise ValueError("Notch frequencies must be positive.")
        nyquist = self.raw.info["sfreq"] / 2
        if (lowpass is not None and lowpass >= nyquist) or any(frequency >= nyquist for frequency in notch):
            raise ValueError(f"Display filter frequencies must be below Nyquist ({nyquist:g} Hz).")
        return highpass, lowpass, notch, bool(self.view_average_reference.get())

    def _display_data(
        self,
        channels: list[str],
        start: float,
        stop: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return display-filtered data from a temporary raw copy, never a derivative."""
        settings = self._display_settings()
        cache_key = (round(start, 6), round(stop, 6), settings)
        if self._display_cache_key != cache_key:
            context_seconds = 5.0
            crop_start = max(0.0, start - context_seconds)
            crop_stop = min(float(self.raw.times[-1]), stop + context_seconds)
            display_raw = self.raw.copy().crop(tmin=crop_start, tmax=crop_stop).load_data()
            highpass, lowpass, notch, average_reference = settings
            # MNE's ``picks='data'`` deliberately excludes auxiliary channels
            # such as EOG.  The display controls should nevertheless clean the
            # channels the reviewer can plot, including EOG.
            display_picks = mne.pick_types(
                display_raw.info,
                eeg=True,
                eog=True,
                exclude=[],
            )
            if highpass is not None or lowpass is not None:
                display_raw.filter(
                    l_freq=highpass,
                    h_freq=lowpass,
                    picks=display_picks,
                    method="iir",
                    iir_params={"order": 4, "ftype": "butter"},
                    verbose="error",
                )
            if notch:
                display_raw.notch_filter(
                    freqs=list(notch),
                    picks=display_picks,
                    method="iir",
                    iir_params={"order": 4, "ftype": "butter"},
                    verbose="error",
                )
            if average_reference:
                display_raw.set_eeg_reference(ref_channels="average", projection=False, verbose="error")
            absolute_times = display_raw.times + crop_start
            keep = (absolute_times >= start) & (absolute_times <= stop)
            self._display_cache_key = cache_key
            self._display_cache = (display_raw, absolute_times, keep)
        display_raw, absolute_times, keep = self._display_cache
        return display_raw.get_data(picks=channels)[:, keep] * 1e6, absolute_times[keep]

    def _reviewer(self) -> str:
        return self.reviewer.get().strip()

    def set_annotation_decision(self, decision: str) -> None:
        indices = self._selected_indices(self.annotation_tree, "a")
        if not indices:
            self.status.set("Select one or more annotations first.")
            return
        timestamp = _timestamp()
        manual_indices = self.annotations.index[
            self.annotations["annotation_source"].eq("manual_qc_review")
        ].intersection(indices)
        retained_indices = list(indices)
        if decision == "dismissed" and len(manual_indices):
            self.annotations = self.annotations.drop(index=manual_indices)
            retained_indices = [index for index in indices if index not in manual_indices]
        if retained_indices:
            self.annotations.loc[retained_indices, "review_status"] = decision
            descriptions = (
                self.annotation_bad_label.get()
                if decision == "approved_bad"
                else self.annotations.loc[retained_indices, "trial_type"]
            )
            self.annotations.loc[retained_indices, "reviewed_description"] = descriptions
            self.annotations.loc[retained_indices, "reviewer"] = self._reviewer()
            self.annotations.loc[retained_indices, "reviewed_at"] = timestamp
            self.annotations.loc[retained_indices, "review_notes"] = self.review_notes.get().strip()
        self.save_current()
        self.refresh_tables(annotation_selection=retained_indices)

    def update_annotation_span(self) -> None:
        index = self._selected_index(self.annotation_tree, "a")
        if index is None:
            self.status.set("Select exactly one annotation to update its span.")
            return
        if len(self._selected_indices(self.annotation_tree, "a")) != 1:
            self.status.set("Select exactly one annotation to update its span.")
            return
        onset, duration = float(self.annotation_onset.get()), float(self.annotation_duration.get())
        if onset < 0 or duration <= 0:
            self.status.set("Onset must be non-negative and duration must be positive.")
            return
        self.annotations.loc[index, ["onset", "duration"]] = [onset, duration]
        self.annotations.loc[index, "review_notes"] = self.review_notes.get().strip()
        self.save_current()
        self.refresh_tables(annotation_selection=[index])

    def add_bad_annotation(self) -> None:
        onset, duration = float(self.annotation_onset.get()), float(self.annotation_duration.get())
        if onset < 0 or duration <= 0:
            self.status.set("Onset must be non-negative and duration must be positive.")
            return
        new_index = 0 if self.annotations.empty else int(self.annotations.index.max()) + 1
        self.annotations.loc[new_index] = {
            "onset": onset,
            "duration": duration,
            "trial_type": "MANUAL_CANDIDATE",
            "annotation_source": "manual_qc_review",
            "review_status": "approved_bad",
            "reviewed_description": self.annotation_bad_label.get(),
            "reviewer": self._reviewer(),
            "reviewed_at": _timestamp(),
            "review_notes": self.review_notes.get().strip(),
        }
        self.save_current()
        self.refresh_tables(annotation_selection=[new_index])

    def add_blink_annotation(self) -> None:
        """Add a reviewer-marked blink without excluding data as BAD."""
        centre = float(self.annotation_onset.get())
        duration = 0.75
        onset = centre - duration / 2
        if onset < 0:
            self.status.set("Manual BLINK centre must be at least 0.375 s.")
            return
        new_index = 0 if self.annotations.empty else int(self.annotations.index.max()) + 1
        self.annotations.loc[new_index] = {
            "onset": onset,
            "duration": duration,
            "trial_type": "MANUAL_BLINK",
            "annotation_source": "manual_qc_review",
            "review_status": "keep",
            "reviewed_description": "BLINK",
            "reviewer": self._reviewer(),
            "reviewed_at": _timestamp(),
            "review_notes": self.review_notes.get().strip(),
        }
        self.save_current()
        self.refresh_tables(annotation_selection=[new_index])

    def set_channel_decision(self, decision: str) -> None:
        indices = self._selected_indices(self.channel_tree, "c")
        if not indices:
            self.status.set("Select one or more channels first.")
            return
        self.channels.loc[indices, "manual_decision"] = decision
        self.channels.loc[indices, "final_decision"] = decision
        self.channels.loc[indices, "reviewer"] = self._reviewer()
        self.channels.loc[indices, "reviewed_at"] = _timestamp()
        self.channels.loc[indices, "review_notes"] = self.review_notes.get().strip()
        self.save_current()
        self.refresh_tables(channel_selection=indices)

    def save_current(self) -> None:
        if self.current_record is None:
            return
        write_review_tables(self.current_record, self.annotations, self.channels)
        self.status.set(f"Saved {self.current_record.label}")

    def close(self) -> None:
        self.save_current()
        self.tk._default_root.destroy()
