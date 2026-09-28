"""Review AMICA or Extended Infomax components in Tk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from spellitout.blink_rate import phase_intervals
from spellitout.ica import (
    AUTO_ARTIFACT_THRESHOLD,
    AUTO_BRAIN_THRESHOLD,
    AUTO_CHANNEL_REVIEW_THRESHOLD,
    AUTO_CONTRIBUTION_THRESHOLD,
    automatic_component_decisions,
)
from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_subject_id


ICLABEL_CLASSES = (
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
)
MAX_OVERVIEW_POINTS = 10_000
MAX_ERP_IMAGE_TIME_POINTS = 1_000
PHASE_STYLES = {
    "listening": ("Listening", "tab:blue"),
    "repetition": ("Repetition", "tab:green"),
    "dots": ("Dots", "tab:orange"),
    "syllabification": ("Syllabification", "tab:purple"),
}


@dataclass(frozen=True)
class EvokedDisplayCache:
    """Precomputed full-resolution mean and compact trial image for one event."""

    times: np.ndarray
    averages: np.ndarray
    images: np.ndarray
    n_epochs: int


def merged_annotation_intervals(
    raw: mne.io.BaseRaw,
    description: str,
    merge_gap_seconds: float = 1.0,
) -> list[tuple[float, float]]:
    """Merge consecutive annotations of one kind into phase intervals."""
    events = sorted(
        (float(onset), float(onset + duration))
        for onset, duration, label in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
        if str(label) == description
    )
    if not events:
        return []
    merged: list[tuple[float, float]] = []
    start, stop = events[0]
    for event_start, event_stop in events[1:]:
        if event_start - stop <= merge_gap_seconds:
            stop = max(stop, event_stop)
        else:
            merged.append((start, stop))
            start, stop = event_start, event_stop
    merged.append((start, stop))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--decomposition",
        choices=("amica", "infomax"),
        default="infomax",
        help="Stage 4 decomposition to review (default: infomax).",
    )
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--reviewer", default="")
    return parser.parse_args()


class ICAReviewApplication:
    def __init__(
        self,
        root,
        raw,
        ica,
        table: pd.DataFrame,
        table_path: Path,
        reviewer: str,
        decomposition: str = "AMICA",
    ):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk, self.ttk = tk, ttk
        self.root, self.raw, self.ica = root, raw, ica
        self.table, self.table_path = table, table_path
        self.reviewer = tk.StringVar(value=reviewer)
        self.notes = tk.StringVar(value="")
        self.start = tk.DoubleVar(value=0.0)
        self.window = tk.DoubleVar(value=15.0)
        self.trace_view = tk.StringVar(value="continuous")
        self.trace_limit = tk.DoubleVar(value=1.0)
        self.status = tk.StringVar(value="")
        self.sources = self.ica.get_sources(self.raw)
        self.evoked_cache: dict[str, EvokedDisplayCache] = {}
        self.evoked_errors: dict[str, str] = {}
        self.active_component: int | None = None
        self.component_trace_limits: dict[int, float] = {}
        self.component_overviews: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.active_source_component: int | None = None
        self.active_source_data: np.ndarray | None = None
        self.slider_after_id = None
        self.trace_axis = None
        self.trace_line = None
        self.overview_window = None
        phases, _ = phase_intervals(self.raw)
        self.phase_intervals = {
            "listening": phases["sentence_listening"],
            "repetition": phases["sentence_repetition"],
            "dots": merged_annotation_intervals(
                self.raw, "pre_syllabification_dot"
            ),
            "syllabification": phases["syllabification"],
        }
        self.phase_artists = []
        self.psd_frequencies, self.component_psd = self._precompute_psd()
        self._precompute_evokeds()
        root.title(f"Spell-IT-OUT {decomposition} component review")
        root.geometry("1300x850")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        controls = ttk.Frame(root, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Reviewer").grid(row=0, column=0)
        ttk.Entry(controls, textvariable=self.reviewer, width=20).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(controls, text="Start (s)").grid(row=0, column=2)
        start_entry = ttk.Entry(controls, textvariable=self.start, width=8)
        start_entry.grid(row=0, column=3, padx=4)
        start_entry.bind("<Return>", lambda _event: self.update_continuous_window())
        start_entry.bind("<FocusOut>", lambda _event: self.update_continuous_window())
        ttk.Label(controls, text="Window (s)").grid(row=0, column=4)
        window_spinbox = ttk.Spinbox(
            controls, from_=2, to=120, textvariable=self.window, width=7,
            command=self.update_continuous_window,
        )
        window_spinbox.grid(row=0, column=5, padx=4)
        window_spinbox.bind("<Return>", lambda _event: self.update_continuous_window())
        window_spinbox.bind("<FocusOut>", lambda _event: self.update_continuous_window())
        ttk.Button(controls, text="◀", command=lambda: self.shift(-1)).grid(row=0, column=6)
        ttk.Button(controls, text="▶", command=lambda: self.shift(1)).grid(row=0, column=7, padx=3)
        ttk.Button(controls, text="Save now", command=self.save).grid(row=0, column=8, padx=(12, 0))
        ttk.Label(controls, text="Trace").grid(row=0, column=9, padx=(16, 0))
        trace = ttk.Combobox(
            controls,
            textvariable=self.trace_view,
            values=("continuous", "evoked_trial", "evoked_syllable"),
            state="readonly",
            width=17,
        )
        trace.grid(row=0, column=10, padx=4)
        trace.bind("<<ComboboxSelected>>", self.trace_view_changed)
        ttk.Label(controls, text="Continuous limit ±").grid(
            row=0, column=11, padx=(12, 0)
        )
        self.trace_limit_spinbox = ttk.Spinbox(
            controls,
            from_=1e-6,
            to=1e12,
            increment=0.1,
            textvariable=self.trace_limit,
            width=10,
            command=self.update_continuous_window,
        )
        self.trace_limit_spinbox.grid(row=0, column=12, padx=4)
        self.trace_limit_spinbox.bind(
            "<Return>", lambda _event: self.update_continuous_window()
        )
        self.trace_limit_spinbox.bind(
            "<FocusOut>", lambda _event: self.update_continuous_window()
        )
        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=2)
        main.add(right, weight=3)
        left.columnconfigure(0, weight=1); left.rowconfigure(0, weight=1)
        cols = (
            "component", "variance_contribution_good_samples", "predicted_label",
            "predicted_probability", "final_decision",
        )
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="extended")
        heading_labels = {
            "variance_contribution_good_samples": "clean contribution",
        }
        for column, width in zip(cols, (72, 125, 112, 102, 115)):
            self.tree.heading(
                column,
                text=heading_labels.get(column, column.replace("_", " ")),
                command=lambda selected_column=column: self.sort_by(selected_column),
            )
            self.tree.column(column, width=width)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.select)
        actions = ttk.Frame(left); actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Keep (K)", command=lambda: self.decide("keep")).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Reject (R)", command=lambda: self.decide("reject")).grid(row=0, column=1, padx=3, sticky="ew")
        ttk.Button(actions, text="Clear (C)", command=lambda: self.decide("unreviewed")).grid(row=0, column=2, sticky="ew")
        ttk.Button(
            actions,
            text="Auto classify (fixed policy)",
            command=self.auto_classify,
        ).grid(
            row=0, column=3, columnspan=3, padx=(10, 0), sticky="ew"
        )
        ttk.Label(
            actions,
            text=(
                f"Keep: brain ≥ {AUTO_BRAIN_THRESHOLD:.0%} or clean contribution "
                f"< {AUTO_CONTRIBUTION_THRESHOLD:.1%}. Reject: muscle/blink/heart/line "
                f"≥ {AUTO_ARTIFACT_THRESHOLD:.0%}. Channel noise ≥ "
                f"{AUTO_CHANNEL_REVIEW_THRESHOLD:.0%}: manual review."
            ),
            wraplength=600,
            justify="left",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(5, 0))
        for column in range(6): actions.columnconfigure(column, weight=1)
        ttk.Label(left, text="Notes").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(left, textvariable=self.notes).grid(row=3, column=0, sticky="ew")
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(8, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.overview_axis = None
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        slider_frame = ttk.Frame(right, padding=(4, 2, 4, 0))
        slider_frame.grid(row=1, column=0, sticky="ew")
        slider_frame.columnconfigure(1, weight=1)
        ttk.Label(slider_frame, text="Continuous position").grid(row=0, column=0, padx=(0, 6))
        self.start_slider = ttk.Scale(
            slider_frame,
            from_=0.0,
            to=max(0.0, float(self.raw.times[-1]) - float(self.window.get())),
            variable=self.start,
            command=self.schedule_slider_draw,
        )
        self.start_slider.grid(row=0, column=1, sticky="ew")
        ttk.Label(root, textvariable=self.status, anchor="w", padding=(8, 0, 8, 8)).grid(row=2, column=0, sticky="ew")
        root.bind("<Control-s>", lambda _event: self.save())
        self.sort_column = "component"
        self.sort_reverse = False
        self.bind_global_shortcuts(root)
        self.refresh()

    def trace_view_changed(self, _event=None) -> None:
        """Enable continuous navigation only when the continuous trace is shown."""
        state = "normal" if self.trace_view.get() == "continuous" else "disabled"
        self.start_slider.configure(state=state)
        self.draw()

    def schedule_slider_draw(self, _value=None) -> None:
        """Debounce redraws while the continuous-position slider is dragged."""
        if self.slider_after_id is not None:
            self.root.after_cancel(self.slider_after_id)
        self.slider_after_id = self.root.after(30, self._draw_from_slider)

    def _draw_from_slider(self) -> None:
        self.slider_after_id = None
        self.update_continuous_window()

    def on_plot_click(self, event) -> None:
        """Centre the continuous window on a time clicked in the overview."""
        if (
            self.trace_view.get() != "continuous"
            or self.overview_axis is None
            or event.inaxes is not self.overview_axis
            or event.xdata is None
        ):
            return
        duration = float(self.raw.times[-1])
        window = max(0.1, float(self.window.get()))
        start = float(event.xdata) - window / 2.0
        self.start.set(min(max(0.0, start), max(0.0, duration - window)))
        self.update_continuous_window()

    def bind_global_shortcuts(self, root) -> None:
        """Bind trace navigation keys before focused widgets consume them."""
        actions = {
            "<KeyPress-k>": lambda: self.decide("keep"),
            "<KeyPress-K>": lambda: self.decide("keep"),
            "<KeyPress-r>": lambda: self.decide("reject"),
            "<KeyPress-R>": lambda: self.decide("reject"),
            "<KeyPress-c>": lambda: self.decide("unreviewed"),
            "<KeyPress-C>": lambda: self.decide("unreviewed"),
            "<Left>": lambda: self.shift(-1),
            "<Right>": lambda: self.shift(1),
        }
        shortcut_tag = "SpellItOutICAShortcuts"
        for sequence, action in actions.items():
            handler = lambda event, callback=action: self.run_shortcut(event, callback)
            root.bind_all(sequence, handler, add="+")
            root.bind_class(shortcut_tag, sequence, handler, add="+")

        def add_shortcut_tag(widget) -> None:
            tags = widget.bindtags()
            if shortcut_tag not in tags:
                widget.bindtags((shortcut_tag, *tags))
            for child in widget.winfo_children():
                add_shortcut_tag(child)

        add_shortcut_tag(root)

    @staticmethod
    def run_shortcut(event, action):
        """Execute a shortcut unless the user is editing a text control."""
        if event.widget.winfo_class() in {
            "Entry", "Spinbox", "Text", "TEntry", "TSpinbox", "TCombobox",
        }:
            return None
        if event.state & 0x4:
            return None
        action()
        return "break"

    def _precompute_psd(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute full-recording Welch PSDs once, rather than per visible window."""
        fmax = min(100.0, self.sources.info["sfreq"] / 2)
        spectrum = self.sources.compute_psd(
            method="welch",
            picks='all',
            fmin=1.0,
            fmax=fmax,
            # n_fft=min(4096, self.sources.n_times),
            # n_overlap=min(2048, max(0, self.sources.n_times - 1)),
            reject_by_annotation="omit",
            verbose="error",
        )
        return spectrum.freqs, spectrum.get_data()

    def _component_evoked(
        self, component: int, event_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if event_name in self.evoked_cache:
            cached = self.evoked_cache[event_name]
            return (
                cached.times,
                cached.averages[component],
                cached.images[component],
                cached.n_epochs,
            )
        if event_name in self.evoked_errors:
            raise RuntimeError(self.evoked_errors[event_name])
        raise RuntimeError(f"No precomputed evoked is available for {event_name!r}.")

    def _precompute_evokeds(self) -> None:
        """Epoch and average each event once for all components during loading."""
        windows = {
            "syllabification_cue": ((0.0, 0.6), None),
            "repetition_cue": ((-4.0, 13.0), (-3.0, 0.0)),
        }
        descriptions = set(self.sources.annotations.description)
        for event_name, ((tmin, tmax), baseline) in windows.items():
            if event_name not in descriptions:
                self.evoked_errors[event_name] = f"No {event_name!r} annotations are available in the ICA-input raw."
                continue
            events, event_ids = mne.events_from_annotations(
                self.sources,
                event_id={event_name: 1},
                verbose="error",
            )
            if event_name not in event_ids or len(events) == 0:
                self.evoked_errors[event_name] = f"No {event_name!r} annotations are available in the ICA-input raw."
                continue
            epochs = mne.Epochs(
                self.sources,
                events,
                event_id=event_ids,
                tmin=tmin,
                tmax=tmax,
                picks="all",
                baseline=baseline,
                preload=True,
                reject_by_annotation=True,
                verbose="error",
            )
            if not len(epochs):
                self.evoked_errors[event_name] = f"No usable {event_name!r} epochs remain after BAD-span exclusion."
                continue
            epoch_data = epochs.get_data(copy=False)
            image_stride = max(
                1,
                int(np.ceil(epoch_data.shape[-1] / MAX_ERP_IMAGE_TIME_POINTS)),
            )
            self.evoked_cache[event_name] = EvokedDisplayCache(
                times=epochs.times.copy(),
                averages=epoch_data.mean(axis=0),
                images=np.transpose(
                    epoch_data[:, :, ::image_stride], (1, 0, 2)
                ).astype(np.float32, copy=True),
                n_epochs=len(epochs),
            )

    def selected(self) -> list[int]:
        return [int(item[1:]) for item in self.tree.selection()]

    def _default_trace_limit(self, component: int) -> float:
        """Estimate a robust symmetric display limit for one complete IC source."""
        source = self._load_active_source(component)
        stride = max(1, int(np.ceil(len(source) / MAX_OVERVIEW_POINTS)))
        overview_data = source[::stride]
        overview_times = np.arange(0, len(source), stride) / self.sources.info["sfreq"]
        self.component_overviews[component] = (overview_times, overview_data)
        finite = source[np.isfinite(source)]
        if not len(finite):
            return 1.0
        centred = finite - np.median(finite)
        limit = float(np.percentile(np.abs(centred), 99.5) * 1.10)
        if not np.isfinite(limit) or limit <= 0:
            limit = float(np.max(np.abs(finite)))
        if not np.isfinite(limit) or limit <= 0:
            return 1.0
        magnitude = 10.0 ** np.floor(np.log10(limit))
        return float(np.ceil(limit / magnitude * 4.0) / 4.0 * magnitude)

    def _load_active_source(self, component: int) -> np.ndarray:
        """Keep only the complete source of the currently displayed component."""
        if self.active_source_component != component or self.active_source_data is None:
            self.active_source_data = self.sources.get_data(picks=[component])[0]
            self.active_source_component = component
        return self.active_source_data

    def _remember_trace_limit(self, component: int | None) -> None:
        if component is None:
            return
        try:
            limit = float(self.trace_limit.get())
        except (TypeError, ValueError):
            return
        if np.isfinite(limit) and limit > 0:
            self.component_trace_limits[component] = limit

    def _load_trace_limit(self, component: int) -> None:
        self._load_active_source(component)
        limit = self.component_trace_limits.get(component)
        if limit is None:
            limit = self._default_trace_limit(component)
            self.component_trace_limits[component] = limit
        elif component not in self.component_overviews:
            self._default_trace_limit(component)
        self.trace_limit.set(limit)
        self.trace_limit_spinbox.configure(increment=max(limit / 20.0, 1e-6))

    def focused_component(self) -> int | None:
        """Return the component the reviewer most recently clicked.

        A Treeview in ``extended`` mode can retain several selections.  Tk
        returns those items in display order, not click order, so using the
        first selected item could keep plotting component 0 while a later row
        was clicked.  ``focus`` tracks the active row instead.
        """
        focused = self.tree.focus()
        if focused.startswith("c"):
            return int(focused[1:])
        selected = self.selected()
        return selected[-1] if selected else None

    def sort_by(self, column: str) -> None:
        """Sort the component table by a clicked heading."""
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        selected = self.selected()
        self.refresh(selected)

    def auto_classify(self) -> None:
        """Replace decisions using the fixed reviewer-independent policy."""
        from tkinter import messagebox

        reviewed = int(self.table.final_decision.isin(("keep", "reject")).sum())
        if not messagebox.askyesno(
            "Apply automatic ICA policy",
            "This will replace all current component decisions. Components that do "
            "not satisfy the fixed automatic criteria will become unreviewed.\n\n"
            f"Current reviewed decisions: {reviewed}. Continue?",
            parent=self.root,
        ):
            return
        decisions = automatic_component_decisions(self.table)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        automatic = decisions.decision_source.eq("automatic")
        self.table["manual_decision"] = "unreviewed"
        self.table["final_decision"] = decisions["final_decision"]
        self.table["decision_source"] = decisions["decision_source"]
        self.table["decision_rule"] = decisions["decision_rule"]
        self.table["reviewer"] = ""
        self.table["reviewed_at"] = ""
        self.table["review_notes"] = ""
        self.table.loc[automatic, "reviewed_at"] = timestamp
        self.table.loc[automatic, "review_notes"] = decisions.loc[
            automatic, "decision_rule"
        ]
        n_keep = int(decisions.final_decision.eq("keep").sum())
        n_reject = int(decisions.final_decision.eq("reject").sum())
        n_unreviewed = int(decisions.final_decision.eq("unreviewed").sum())
        self.save()
        selected = self.selected()
        self.refresh(selected)
        self.status.set(
            f"Fixed policy applied: {n_keep} keep, {n_reject} reject, "
            f"{n_unreviewed} unreviewed."
        )

    def refresh(self, selected: list[int] | None = None) -> None:
        if selected is None: selected = self.selected()
        self.tree.delete(*self.tree.get_children())
        display_table = self.table
        if self.sort_column in {
            "component", "variance_contribution_good_samples", "predicted_probability",
        }:
            values = pd.to_numeric(display_table[self.sort_column], errors="coerce")
            display_table = display_table.assign(_sort_value=values).sort_values(
                "_sort_value", ascending=not self.sort_reverse, kind="stable"
            )
        else:
            display_table = display_table.sort_values(
                self.sort_column, ascending=not self.sort_reverse, kind="stable"
            )
        for _, row in display_table.iterrows():
            component = int(row.component)
            ratio = float(row.variance_contribution_good_samples)
            probability = float(row.predicted_probability)
            self.tree.insert(
                "", "end", iid=f"c{component}",
                values=(
                    component,
                    f"{ratio:.2%}",
                    row.predicted_label,
                    f"{probability:.1%}",
                    row.final_decision,
                ),
            )
        valid = [f"c{component}" for component in selected if component in set(self.table.component)]
        if valid:
            self.tree.selection_set(valid)
            self.tree.focus(valid[-1])
        elif len(self.table):
            self.tree.selection_set("c0")
            self.tree.focus("c0")
        self.select()

    def select(self, _event=None) -> None:
        component = self.focused_component()
        if component is None:
            return
        if component != self.active_component:
            self._remember_trace_limit(self.active_component)
            self._load_trace_limit(component)
        self.active_component = component
        row = self.table.set_index("component").loc[component]
        self.notes.set(str(row.review_notes))
        self.draw()

    def decide(self, decision: str) -> None:
        selected = self.selected()
        if not selected:
            self.status.set("Select one or more components first."); return
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        mask = self.table.component.isin(selected)
        self.table.loc[mask, "manual_decision"] = decision
        self.table.loc[mask, "final_decision"] = decision
        self.table.loc[mask, "decision_source"] = (
            "unreviewed" if decision == "unreviewed" else "manual"
        )
        self.table.loc[mask, "decision_rule"] = ""
        if decision == "unreviewed":
            self.table.loc[mask, ["reviewer", "reviewed_at", "review_notes"]] = ""
        else:
            self.table.loc[mask, "reviewer"] = self.reviewer.get().strip()
            self.table.loc[mask, "reviewed_at"] = timestamp
            self.table.loc[mask, "review_notes"] = self.notes.get().strip()
        self.save(); self.refresh(selected)
        # highlight the next component after the last selected one, if any remain
        next_component = min((c for c in self.table.component if c > max(selected)), default=None)
        if next_component is not None:
            self.tree.selection_set(f"c{next_component}")
            self.tree.focus(f"c{next_component}")
            self.select()

    def shift(self, direction: int) -> None:
        if self.trace_view.get() != "continuous":
            return
        duration = float(self.raw.times[-1])
        window = max(0.1, float(self.window.get()))
        start = self.start.get() + direction * window * 0.8
        self.start.set(min(max(0.0, start), max(0.0, duration - window)))
        self.update_continuous_window()

    def _continuous_window_data(
        self, component: int
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Return the bounded visible slice without recomputing ICA sources."""
        duration = float(self.raw.times[-1])
        window = max(0.1, float(self.window.get()))
        start = min(
            max(0.0, float(self.start.get())), max(0.0, duration - window)
        )
        stop = min(duration, start + window)
        if abs(float(self.start.get()) - start) > 1e-9:
            self.start.set(start)
        self.start_slider.configure(to=max(0.0, duration - window))
        sfreq = float(self.raw.info["sfreq"])
        first = int(start * sfreq)
        last = int(stop * sfreq)
        source = self._load_active_source(component)[first:last]
        times = np.arange(first, first + len(source)) / sfreq
        return start, stop, times, source

    def _validated_trace_limit(self, component: int) -> float:
        try:
            trace_limit = float(self.trace_limit.get())
        except (TypeError, ValueError):
            trace_limit = self.component_trace_limits.get(component, 1.0)
        if not np.isfinite(trace_limit) or trace_limit <= 0:
            trace_limit = self.component_trace_limits.get(component, 1.0)
        self.trace_limit.set(trace_limit)
        self.component_trace_limits[component] = trace_limit
        return trace_limit

    def _update_phase_markers(self, start: float, stop: float) -> None:
        """Show phase spans overlapping the current continuous window."""
        for artist in self.phase_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self.phase_artists = []
        if self.trace_axis is None:
            return

        for phase, intervals in self.phase_intervals.items():
            label, color = PHASE_STYLES[phase]
            for phase_start, phase_stop in intervals:
                if phase_stop <= start or phase_start >= stop:
                    continue
                visible_start = max(start, phase_start)
                visible_stop = min(stop, phase_stop)
                span = self.trace_axis.axvspan(
                    visible_start,
                    visible_stop,
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                    zorder=0,
                )
                marker_label = self.trace_axis.text(
                    (visible_start + visible_stop) / 2.0,
                    0.98,
                    label,
                    transform=self.trace_axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    clip_on=True,
                    zorder=3,
                )
                self.phase_artists.extend((span, marker_label))

    @staticmethod
    def _draw_evoked_trial_phases(
        axis, epoch_times: np.ndarray, show_labels: bool = True
    ) -> None:
        """Highlight the fixed task phases relative to repetition cue onset."""
        epoch_start = float(epoch_times[0])
        epoch_stop = float(epoch_times[-1])
        phases = (
            ("listening", epoch_start, 0.0),
            ("repetition", 0.0, 5.0),
            ("dots", 7.0, 8.8),
            ("syllabification", 8.8, epoch_stop),
        )
        for phase, phase_start, phase_stop in phases:
            visible_start = max(epoch_start, phase_start)
            visible_stop = min(epoch_stop, phase_stop)
            if visible_stop <= visible_start:
                continue
            label, color = PHASE_STYLES[phase]
            axis.axvspan(
                visible_start,
                visible_stop,
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=0,
            )
            if show_labels:
                axis.text(
                    (visible_start + visible_stop) / 2.0,
                    0.98,
                    label,
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    clip_on=True,
                    zorder=3,
                )

    def update_continuous_window(self) -> None:
        """Fast path for navigation: update only trace and overview viewport."""
        if self.trace_view.get() != "continuous":
            return
        component = self.active_component
        if component is None or self.trace_axis is None or self.trace_line is None:
            self.draw()
            return
        start, stop, times, source = self._continuous_window_data(component)
        trace_limit = self._validated_trace_limit(component)
        self.trace_line.set_data(times, source)
        self.trace_axis.set_xlim(start, stop)
        self.trace_axis.set_ylim(-trace_limit, trace_limit)
        self._update_phase_markers(start, stop)
        if self.overview_axis is not None:
            self.overview_axis.set_ylim(-trace_limit, trace_limit)
            if self.overview_window is not None:
                self.overview_window.remove()
            self.overview_window = self.overview_axis.axvspan(
                start, stop, color="tab:blue", alpha=0.22, linewidth=0
            )
        self.canvas.draw_idle()

    def draw(self) -> None:
        component = self.active_component
        if component is None:
            component = self.focused_component()
        if component is None:
            return
        duration = float(self.raw.times[-1])
        start, stop, times, sources = self._continuous_window_data(component)
        row = self.table.set_index("component").loc[component]
        self.figure.clear()
        self.phase_artists = []
        grid = self.figure.add_gridspec(
            4, 2, width_ratios=(1.35, 1), height_ratios=(1.2, 1, 0.42, 1)
        )
        topography = self.figure.add_subplot(grid[0, 0])
        probabilities = self.figure.add_subplot(grid[0, 1])
        trace = self.figure.add_subplot(grid[1, :])
        self.trace_axis = trace
        overview = self.figure.add_subplot(grid[2, :])
        self.overview_axis = overview
        spectrum = self.figure.add_subplot(grid[3, :])
        mne.viz.plot_topomap(self.ica.get_components()[:, component], self.raw.info, axes=topography, show=False)
        topography.set_title(
            f"Component {component} topography\n"
            f"Good-sample contribution: "
            f"{float(row.variance_contribution_good_samples):.2%}"
        )
        probabilities.axis("off")
        probability_lines = ["ICLabel probabilities"] + [
            f"{label}: {float(row[label]):.1%}"
            for label in ICLABEL_CLASSES
        ]
        probabilities.text(
            0.02, 0.98, "\n".join(probability_lines), va="top", ha="left",
            family="monospace", fontsize=13,
        )
        trace_view = self.trace_view.get()
        if trace_view == "continuous":
            trace_limit = self._validated_trace_limit(component)
            (self.trace_line,) = trace.plot(times, sources, linewidth=.7)
            trace.set(
                xlabel="Time (s)", ylabel="a.u.", xlim=(start, stop),
                ylim=(-trace_limit, trace_limit), title="Continuous component source",
            )
            self._update_phase_markers(start, stop)
            overview_times, overview_data = self.component_overviews[component]
            overview.plot(overview_times, overview_data, linewidth=0.45, color="0.35")
            self.overview_window = overview.axvspan(
                start, stop, color="tab:blue", alpha=0.22, linewidth=0
            )
            overview.set(
                xlim=(0.0, duration), ylim=(-trace_limit, trace_limit),
                yticks=[],
                # title=(
                #     f"Full component overview (≤ {MAX_OVERVIEW_POINTS:,} points; "
                #     "click to centre window)"
                # ),
            )
            overview.tick_params(axis="x", labelsize=7)
            overview.grid(axis="x", alpha=.18)
        else:
            self.trace_line = None
            self.overview_window = None
            try:
                event_name = (
                    "repetition_cue"
                    if trace_view == "evoked_trial"
                    else "syllabification_cue"
                )
                (
                    epoch_times,
                    evoked,
                    trial_image,
                    n_epochs,
                ) = self._component_evoked(component, event_name)
                finite_image = np.abs(trial_image[np.isfinite(trial_image)])
                color_limit = (
                    float(np.percentile(finite_image, 98.0))
                    if finite_image.size
                    else 1.0
                )
                if not np.isfinite(color_limit) or color_limit <= 0:
                    color_limit = 1.0
                image = trace.imshow(
                    trial_image,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    extent=(
                        float(epoch_times[0]),
                        float(epoch_times[-1]),
                        0.5,
                        n_epochs + 0.5,
                    ),
                    cmap="RdBu_r",
                    vmin=-color_limit,
                    vmax=color_limit,
                    rasterized=True,
                )
                self.figure.colorbar(
                    image,
                    ax=trace,
                    pad=0.01,
                    aspect=25,
                    label="IC amplitude (a.u.)",
                )
                trace.axvline(0, color="black", linestyle="--", linewidth=.8)
                if trace_view == "evoked_trial":
                    self._draw_evoked_trial_phases(trace, epoch_times)
                trace.set(
                    xlabel="", ylabel="Trial (chronological)",
                    xlim=(float(epoch_times[0]), float(epoch_times[-1])),
                    title=f"{event_name} ERP image ({n_epochs} usable epochs)",
                )
                overview.plot(epoch_times, evoked, linewidth=0.9, color="black")
                overview.axvline(0, color="black", linestyle="--", linewidth=.8)
                if trace_view == "evoked_trial":
                    self._draw_evoked_trial_phases(
                        overview, epoch_times, show_labels=False
                    )
                overview.set(
                    xlabel="Time relative to cue (s)",
                    ylabel="Mean\n(a.u.)",
                    xlim=(float(epoch_times[0]), float(epoch_times[-1])),
                    title="Trial average",
                )
                overview.grid(alpha=.25)
            except RuntimeError as error:
                trace.text(.5, .5, str(error), ha="center", va="center", transform=trace.transAxes)
                trace.set_axis_off()
                overview.set_axis_off()
        if trace_view == "continuous":
            trace.grid(alpha=.25)
        spectrum.semilogy(self.psd_frequencies, self.component_psd[component], linewidth=1)
        spectrum.set(xlim=(1, min(100, self.raw.info["sfreq"]/2)), xlabel="Frequency (Hz)", ylabel="PSD (a.u.²/Hz)", title="Full-recording component PSD")
        spectrum.grid(alpha=.25)
        self.canvas.draw_idle()

    def save(self) -> None:
        self.table.to_csv(self.table_path, sep="\t", index=False)
        self.status.set(f"Saved {self.table_path.name}")

    def close(self) -> None:
        self.save(); self.root.destroy()


def main(args: argparse.Namespace) -> None:
    import tkinter as tk
    display_name = "AMICA" if args.decomposition == "amica" else "Extended Infomax"
    register_derivative_description(
        args.derivatives_path,
        f"{args.decomposition}-components",
        f"Fixed-policy automatic classification followed by human {display_name} "
        f"component review; decisions and their source are stored in "
        f"desc-{args.decomposition}_components.tsv.",
    )
    subject = get_subject_id(args.subject)
    base = f"{subject}_ses-01_task-{args.condition}"
    eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
    input_desc = "ica-input" if args.decomposition == "amica" else "infomax-input"
    raw = mne.io.read_raw_fif(
        eeg_dir / f"{base}_desc-{input_desc}_raw.fif", preload=False, verbose="error"
    )
    ica = mne.preprocessing.read_ica(
        eeg_dir / f"{base}_desc-{args.decomposition}_ica.fif", verbose="error"
    )
    table_path = eeg_dir / f"{base}_desc-{args.decomposition}_components.tsv"
    table = pd.read_csv(table_path, sep="\t")
    required = (
        "variance_contribution",
        "variance_contribution_good_samples",
        "predicted_label",
        "predicted_probability",
        *ICLABEL_CLASSES,
    )
    missing = [column for column in required if column not in table]
    if missing:
        raise RuntimeError(
            f"This {display_name} component table predates ICLabel evidence. Re-run "
            f"{'04_amica.py' if args.decomposition == 'amica' else '04_ica.py'} "
            "with --overwrite to generate: " + ", ".join(missing)
        )
    table = table.fillna("")
    root = tk.Tk()
    ICAReviewApplication(
        root, raw, ica, table, table_path, args.reviewer, display_name
    )
    root.mainloop()


if __name__ == "__main__":
    main(parse_args())
