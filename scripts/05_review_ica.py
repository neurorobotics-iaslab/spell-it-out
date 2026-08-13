"""Review AMICA components and save manual keep/reject decisions in Tk."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_subject_id


ICLABEL_CLASSES = (
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--reviewer", default="")
    return parser.parse_args()


class ICAReviewApplication:
    def __init__(self, root, raw, ica, table: pd.DataFrame, table_path: Path, reviewer: str):
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
        self.view_mode = tk.StringVar(value="Continuous")
        self.evoked_event = tk.StringVar(value="syllable_cue")
        self.status = tk.StringVar(value="")
        self.sources = self.ica.get_sources(self.raw)
        self.evoked_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, int]] = {}
        self.evoked_errors: dict[str, str] = {}
        self.psd_frequencies, self.component_psd = self._precompute_psd()
        self._precompute_evokeds()
        root.title("Spell-IT-OUT AMICA component review")
        root.geometry("1300x850")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        controls = ttk.Frame(root, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Reviewer").grid(row=0, column=0)
        ttk.Entry(controls, textvariable=self.reviewer, width=20).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(controls, text="Start (s)").grid(row=0, column=2)
        ttk.Entry(controls, textvariable=self.start, width=8).grid(row=0, column=3, padx=4)
        ttk.Label(controls, text="Window (s)").grid(row=0, column=4)
        ttk.Spinbox(controls, from_=2, to=120, textvariable=self.window, width=7, command=self.draw).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="◀", command=lambda: self.shift(-1)).grid(row=0, column=6)
        ttk.Button(controls, text="▶", command=lambda: self.shift(1)).grid(row=0, column=7, padx=3)
        ttk.Button(controls, text="Save now", command=self.save).grid(row=0, column=8, padx=(12, 0))
        ttk.Label(controls, text="View").grid(row=0, column=9, padx=(16, 0))
        mode = ttk.Combobox(controls, textvariable=self.view_mode, values=("Continuous", "Evoked"), state="readonly", width=11)
        mode.grid(row=0, column=10, padx=4)
        mode.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        ttk.Label(controls, text="Event").grid(row=0, column=11)
        event = ttk.Combobox(controls, textvariable=self.evoked_event, values=("syllable_cue", "repetition_cue"), state="readonly", width=16)
        event.grid(row=0, column=12, padx=4)
        event.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=2)
        main.add(right, weight=3)
        left.columnconfigure(0, weight=1); left.rowconfigure(0, weight=1)
        cols = (
            "component", "variance_contribution", "predicted_label",
            "predicted_probability", "final_decision",
        )
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="extended")
        for column, width in zip(cols, (72, 100, 112, 102, 115)):
            self.tree.heading(column, text=column.replace("_", " "))
            self.tree.column(column, width=width)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.select)
        actions = ttk.Frame(left); actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Keep", command=lambda: self.decide("keep")).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Reject", command=lambda: self.decide("reject")).grid(row=0, column=1, padx=3, sticky="ew")
        ttk.Button(actions, text="Clear", command=lambda: self.decide("unreviewed")).grid(row=0, column=2, sticky="ew")
        for column in range(3): actions.columnconfigure(column, weight=1)
        ttk.Label(left, text="Notes").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(left, textvariable=self.notes).grid(row=3, column=0, sticky="ew")
        right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(8, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        ttk.Label(root, textvariable=self.status, anchor="w", padding=(8, 0, 8, 8)).grid(row=2, column=0, sticky="ew")
        root.bind("<Control-s>", lambda _event: self.save())
        self.refresh()

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

    def _component_evoked(self, component: int, event_name: str) -> tuple[np.ndarray, np.ndarray, int]:
        key = (component, event_name)
        if key in self.evoked_cache:
            return self.evoked_cache[key]
        if event_name in self.evoked_errors:
            raise RuntimeError(self.evoked_errors[event_name])
        raise RuntimeError(f"No precomputed evoked is available for {event_name!r}.")

    def _precompute_evokeds(self) -> None:
        """Epoch and average each event once for all components during loading."""
        windows = {
            "syllable_cue": ((0.0, 0.6), None),
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
            evoked = epochs.average(picks='all')
            for component in range(evoked.data.shape[0]):
                self.evoked_cache[(component, event_name)] = (
                    evoked.times,
                    evoked.data[component],
                    len(epochs),
                )

    def selected(self) -> list[int]:
        return [int(item[1:]) for item in self.tree.selection()]

    def refresh(self, selected: list[int] | None = None) -> None:
        if selected is None: selected = self.selected()
        self.tree.delete(*self.tree.get_children())
        for _, row in self.table.iterrows():
            component = int(row.component)
            ratio = float(row.variance_contribution)
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
        if valid: self.tree.selection_set(valid)
        elif len(self.table): self.tree.selection_set("c0")
        self.select()

    def select(self, _event=None) -> None:
        selected = self.selected()
        if not selected: return
        row = self.table.set_index("component").loc[selected[0]]
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
        self.table.loc[mask, "reviewer"] = self.reviewer.get().strip()
        self.table.loc[mask, "reviewed_at"] = timestamp
        self.table.loc[mask, "review_notes"] = self.notes.get().strip()
        self.save(); self.refresh(selected)
        # highlight the next component after the last selected one, if any remain
        next_component = min((c for c in self.table.component if c > max(selected)), default=None)
        if next_component is not None:
            self.tree.selection_set(f"c{next_component}"); self.select()

    def shift(self, direction: int) -> None:
        self.start.set(max(0.0, self.start.get() + direction * self.window.get() * .8)); self.draw()

    def draw(self) -> None:
        selected = self.selected()
        if not selected: return
        component = selected[0]
        duration = float(self.raw.times[-1]); window = max(.1, float(self.window.get()))
        start = min(max(0.0, float(self.start.get())), max(0.0, duration - window)); stop = min(duration, start + window)
        sources = self.sources.get_data(picks=[component], start=int(start*self.raw.info["sfreq"]), stop=int(stop*self.raw.info["sfreq"]))[0]
        times = np.arange(len(sources)) / self.raw.info["sfreq"] + start
        row = self.table.set_index("component").loc[component]
        self.figure.clear()
        grid = self.figure.add_gridspec(3, 2, width_ratios=(1.35, 1), height_ratios=(1.2, 1, 1))
        topography = self.figure.add_subplot(grid[0, 0])
        probabilities = self.figure.add_subplot(grid[0, 1])
        trace = self.figure.add_subplot(grid[1, :])
        spectrum = self.figure.add_subplot(grid[2, :])
        mne.viz.plot_topomap(self.ica.get_components()[:, component], self.raw.info, axes=topography, show=False)
        topography.set_title(f"Component {component} topography\nContribution: {float(row.variance_contribution):.2%}")
        probabilities.axis("off")
        probability_lines = ["ICLabel probabilities"] + [
            f"{label}: {float(row[label]):.1%}"
            for label in ICLABEL_CLASSES
        ]
        probabilities.text(
            0.02, 0.98, "\n".join(probability_lines), va="top", ha="left",
            family="monospace", fontsize=9,
        )
        if self.view_mode.get() == "Continuous":
            trace.plot(times, sources, linewidth=.7)
            trace.set(xlabel="Time (s)", ylabel="a.u.", xlim=(start, stop), title="Continuous component source")
        else:
            try:
                epoch_times, evoked, n_epochs = self._component_evoked(component, self.evoked_event.get())
                trace.plot(epoch_times, evoked, linewidth=1)
                trace.axvline(0, color="black", linestyle="--", linewidth=.8)
                trace.set(
                    xlabel="Time relative to cue (s)", ylabel="a.u.",
                    title=f"Average {self.evoked_event.get()} ({n_epochs} usable epochs)",
                )
            except RuntimeError as error:
                trace.text(.5, .5, str(error), ha="center", va="center", transform=trace.transAxes)
                trace.set_axis_off()
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
    register_derivative_description(
        args.derivatives_path,
        "amica-components",
        "Human AMICA component review; decisions are stored in desc-amica_components.tsv.",
    )
    subject = get_subject_id(args.subject)
    base = f"{subject}_ses-01_task-{args.condition}"
    eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
    raw = mne.io.read_raw_fif(eeg_dir / f"{base}_desc-ica-input_raw.fif", preload=False, verbose="error")
    ica = mne.preprocessing.read_ica(eeg_dir / f"{base}_desc-amica_ica.fif", verbose="error")
    table_path = eeg_dir / f"{base}_desc-amica_components.tsv"
    table = pd.read_csv(table_path, sep="\t")
    required = ("variance_contribution", "predicted_label", "predicted_probability", *ICLABEL_CLASSES)
    missing = [column for column in required if column not in table]
    if missing:
        raise RuntimeError(
            "This AMICA component table predates ICLabel evidence. Re-run 04_amica.py "
            "with --overwrite to generate: " + ", ".join(missing)
        )
    table = table.fillna("")
    root = tk.Tk(); ICAReviewApplication(root, raw, ica, table, table_path, args.reviewer); root.mainloop()


if __name__ == "__main__":
    main(parse_args())
