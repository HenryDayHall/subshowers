"""
Plot observables of the subshowers a config describes, as the detector array
it describes sees them. It is like scripts/demo_sampler.py, but takes the
showers, energy cuts and detector from the config.

    python scripts/plot_observables_from_config.py configs/default.yaml

Subshower sets the config asks for that haven't been made, or can't be read,
are listed, and the plots are made from the rest. Everything printed is also
written to report.txt in the output folder.

Each set's detector responses are saved in responses/ in the output folder,
and used again while the set, the detector and the sampling options stay the
same. So plots can be remade quickly, and the dataset can be compared later
with other populations measured with the same array, such as subshowers from
a generative model (see subshowers.responses.Responses).

What the same subshowers put on the ground is summarised too, for the
observables used to compare air shower simulations at ground level (see
subshowers.ground): lateral distributions, energy spectra, particle numbers,
arrival times and readings at a reference distance. The summaries are saved
in ground/, and used again while the set, the sampling options and the
config's ground section stay the same. Distances and delays are measured
about each subshower's own axis, or with --frame shower, about the shower's.

The plots for each energy cut go in e<cut>/, grouped by the primary energy of
the shower and by the particle that starts the subshower. With more than one
cut, all_cuts/ compares the cuts.
"""

import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from subshowers import (  # noqa: E402
    datasets,
    detector,
    generation,
    ground,
    ground_observables,
    observables,
)
from subshowers.ground import GroundSettings, GroundSummaries  # noqa: E402
from subshowers.responses import Responses  # noqa: E402
from subshowers.sample_plotting import show_subshower  # noqa: E402
from subshowers.sampling import SubshowerSampler  # noqa: E402

DPI = 130
N_EXAMPLES = 4
ELECTRONS_AND_POSITRONS = [11, -11]
FIGURES = [  # (file name, function, title)
    (
        "A",
        observables.observables_A,
        "Subshower start conditions vs detected response",
    ),
    ("B", observables.observables_B, "Subshower start energy vs deposit shape"),
    ("C", observables.observables_C, "Average per-sample deposit spectra"),
    ("distributions", observables.distributions, "Distributions of observables"),
]
# (file name, title, whether it's measured about an axis, and a function
# making it from (by_group, frame, reference distance, title))
GROUND_FIGURES = [
    (
        "ground_lateral",
        "Lateral distributions on the ground",
        True,
        lambda by_group, frame, distance, title: (
            ground_observables.lateral_distributions(
                by_group, frame, reference_distance=distance, title=title
            )
        ),
    ),
    (
        "ground_spectra",
        "Energy spectra on the ground",
        False,
        lambda by_group, frame, distance, title: (
            ground_observables.energy_spectra(by_group, title=title)
        ),
    ),
    (
        "ground_numbers",
        "Numbers of particles on the ground",
        False,
        lambda by_group, frame, distance, title: (
            ground_observables.particle_numbers(by_group, title=title)
        ),
    ),
    (
        "ground_timing",
        "Arrival times on the ground",
        True,
        lambda by_group, frame, distance, title: (
            ground_observables.arrival_times(by_group, frame, title=title)
        ),
    ),
    (
        "ground_reference",
        "Readings at the reference distance",
        True,
        lambda by_group, frame, distance, title: (
            ground_observables.readings_at_reference(by_group, frame, distance, title)
        ),
    ),
]


class Report:
    """Prints lines, and keeps them to write to report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def items(self, heading, items):
        if items:
            self(heading)
            for item in items:
                self(f"  {item}")

    def write(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config_file")
    parser.add_argument(
        "--output-dir",
        help="folder for the plots (default: plots/<run_name> in the showers folder)",
    )
    parser.add_argument(
        "--max-subshowers",
        type=int,
        metavar="N",
        help="use at most N subshowers from each set, chosen at random (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for choosing subshowers and examples (default: 42)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="measure detector responses again, even where saved ones match",
    )
    parser.add_argument(
        "--detected-only",
        action="store_true",
        help="plot only the subshowers that deposited energy in some detector",
    )
    parser.add_argument(
        "--frame",
        choices=ground.FRAMES,
        default="subshower",
        help="measure ground distances and delays about each subshower's own axis,"
        " or about the shower's (default: subshower)",
    )
    args = parser.parse_args()
    if args.max_subshowers is not None and args.max_subshowers < 1:
        parser.error("--max-subshowers must be at least 1")
    return args


# ----------------------------------------------------------------- helpers


def describe_seconds(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    return generation.describe_minutes(seconds / 60)


def percent(part, whole):
    return f"{part / whole:.1%}" if whole else "-"


def describe_detector(config, array):
    name, arguments = detector.generator_settings(config)
    listed = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({listed}), {len(array)} detectors"


def describe_energy(gev):
    if 0 < gev < 0.1:
        return f"{gev * 1e3:g} MeV"
    return f"{gev:g} GeV"


def describe_ground(settings, frame):
    minimum = ", ".join(
        f"{ground.SPECIES_LABELS[name]} {describe_energy(energy)}"
        for name, energy in settings.min_kinetic_energy.items()
    )
    return (
        f"z = {settings.height:g} m, counting particles above {minimum};"
        f" readings at {settings.reference_distance[frame]:g} m from the"
        f" {frame} axis"
    )


def primary_label(shower):
    """A shower's group: its primary energy, or its folder if that's unknown."""
    if shower.energy_text is not None:
        return f"{shower.energy_text} GeV primary"
    return shower.subfolder


def responses_path(output_dir, subshower_set, folder="responses"):
    name = subshower_set.shower.subfolder.replace("/", "__")
    return os.path.join(output_dir, folder, f"{name}_e{subshower_set.energy_cut}.npz")


def ground_path(output_dir, subshower_set):
    return responses_path(output_dir, subshower_set, folder="ground")


def responses_settings(subshower_set, config, args):
    """What decides a set's responses; saved responses are used if they match."""
    name, arguments = detector.generator_settings(config)
    yaml_path = subshower_set.output[: -len(".h5")] + ".yaml"
    settings = {
        "subshower_file": os.path.abspath(subshower_set.output),
        # Output.save writes the .yaml last, so this changes when a set is remade
        "subshower_file_time": os.path.getmtime(yaml_path),
        "energy_cut": subshower_set.energy_cut,
        "leaves_only": subshower_set.leaves_only,
        "detector": {"name": name, **arguments},
        "max_subshowers": args.max_subshowers,
        "seed": args.seed,
    }
    # as it will read back from the saved file
    return json.loads(json.dumps(settings, default=str))


def ground_settings_for(settings, ground_settings):
    """
    What decides a set's ground summaries: what picks its subshowers (the
    responses' settings, but for the detector) and the ground settings.
    """
    chosen = {key: value for key, value in settings.items() if key != "detector"}
    chosen["ground"] = ground_settings.as_dict()
    return json.loads(json.dumps(chosen, default=str))


def progress(samples, total, label, every=60.0):
    """Pass samples on, printing progress at most every `every` seconds."""
    start = last = time.monotonic()
    for count, sample in enumerate(samples, start=1):
        yield sample
        now = time.monotonic()
        if now - last >= every and count < total:
            print(
                f"    {label}: {count} of {total} subshowers,"
                f" {describe_seconds(now - start)}",
                flush=True,
            )
            last = now


def saved_with(path, settings, kind):
    """
    What a file holds, read as kind (Responses or GroundSummaries), if it was
    saved with these settings; otherwise None.
    """
    if not os.path.isfile(path):
        return None
    try:
        saved = kind.read_metadata(path)
        if saved.get("settings") == settings:
            return kind.load(path)
    except (OSError, ValueError, KeyError):
        pass
    return None


def measure(check, array, settings, path, args, reuse):
    """
    The detector responses to one subshower set, how they were got, and the
    set's DatasetSource if it was opened to measure them (None otherwise):
    the responses are read from path if they were saved with the same
    settings, otherwise measured and saved there. The source is left open,
    for ground summaries, so release it when they're made.
    """
    if reuse:
        saved = saved_with(path, settings, Responses)
        if saved is not None:
            return saved, "saved earlier", None
    start = time.monotonic()
    source = check.source()
    n_in_set = len(source)  # reads the set, and the shower's history
    # not shifted across the ground: the ground observables are of the
    # subshowers where they were simulated, and the responses go with them
    sampler = SubshowerSampler([source], array, seed=args.seed, max_open=1, shift=False)
    n_wanted = n_in_set
    if args.max_subshowers is not None:
        n_wanted = min(n_in_set, args.max_subshowers)
    samples = itertools.islice(sampler.iter_epoch(shuffle=True), n_wanted)
    responses = Responses.from_samples(
        progress(samples, n_wanted, check.label),
        len(array),
        source=check.subshower_set.shower.subfolder,
        metadata={
            "settings": settings,
            "n_subshowers_in_set": n_in_set,
            "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    responses.save(path)
    how = f"measured in {describe_seconds(time.monotonic() - start)}"
    return responses, how, source


def summarise_ground(check, responses, settings, ground_settings, path, reuse, source):
    """
    What the subshowers of responses put on the ground, as GroundSummaries
    of the same subshowers in the same order, how they were got, and the
    set's DatasetSource if it's open: read from path if they were saved with
    the same settings, otherwise worked out (opening the source if it isn't
    open) and saved there.
    """
    if reuse:
        saved = saved_with(path, settings, GroundSummaries)
        if saved is not None:
            try:
                ground.check_aligned(saved, responses)
                return saved, "as worked out before", source
            except ValueError:
                pass  # not the same subshowers, so work them out again
    start = time.monotonic()
    if source is None:
        source = check.source()
    numbers = responses.values["subshower_number"]
    summaries = GroundSummaries.from_source(
        source,
        numbers,
        ground_settings,
        metadata={
            "settings": settings,
            "worked_out": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        progress=lambda chosen: progress(
            chosen, len(numbers), f"{check.label} (ground)"
        ),
    )
    summaries.save(path)
    how = f"worked out in {describe_seconds(time.monotonic() - start)}"
    return summaries, how, source


def coordinate_ranges(responses, array):
    """Where the detectors are, and where the subshowers start."""
    lines = []
    for i, axis in enumerate("xyz"):
        centres = array.centers[:, i]
        starts = responses.values["start_position"][:, i]
        lines.append(
            f"{axis}: detectors {centres.min():.6g} to {centres.max():.6g} m,"
            f" subshower starts {starts.min():.6g} to {starts.max():.6g} m"
        )
    return lines


# ---------------------------------------------------------------- plotting


def save(fig, path, written):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    written.append(path)


def plot_groups(by_group, folder, suffix, context, written, ground_options=None):
    """
    observables A, B, C and distributions of some groups of subshowers, and
    with ground_options, (frame, reference distance), their ground figures.
    """
    if not by_group:
        return
    for name, function, title in FIGURES:
        fig = function(by_group, title=f"{title}\n{context}")
        save(fig, os.path.join(folder, f"{name}_{suffix}.png"), written)
    if ground_options is None:
        return
    frame, distance = ground_options
    for name, title, about_axis, function in GROUND_FIGURES:
        if about_axis:
            title = f"{title}, about the {frame} axis"
            name = f"{name}_{frame}_axis"
        fig = function(by_group, frame, distance, f"{title}\n{context}")
        save(fig, os.path.join(folder, f"{name}_{suffix}.png"), written)


def plot_examples(responses, array, path, context, seed, written):
    """The detectors hit by a few subshowers that deposited energy."""
    detected = np.flatnonzero(responses.detected)
    if not len(detected):
        return
    rng = np.random.default_rng(seed)
    chosen = rng.choice(detected, size=min(N_EXAMPLES, len(detected)), replace=False)
    fig, axarr = plt.subplots(2, 2, figsize=(13, 11))
    for ax, i in zip(axarr.ravel(), chosen):
        sample = responses.sample(i)
        plt.sca(ax)
        with contextlib.redirect_stdout(io.StringIO()):  # it prints as it works
            show_subshower(
                sample.start_kinetic_energy,
                sample.energy_per_detector,
                array=array,
                axis=ax,
                initial_position=sample.start_position,
            )
        if sample.n_leaves >= 0:
            leaves = f"{sample.n_detected_leaves}/{sample.n_leaves} leaves detected, "
        else:
            leaves = ""
        n_hit = sample.n_hit_detectors
        ax.set_title(
            f"{sample.source}, pdg {sample.start_pdg},"
            f" start KE {sample.start_kinetic_energy:.3g} GeV\n"
            f"{leaves}{n_hit} detector{'' if n_hit == 1 else 's'} hit,"
            f" E_det {sample.total_detected_energy:.3g} GeV",
            fontsize=9,
        )
    for ax in axarr.ravel()[len(chosen) :]:
        ax.axis("off")
    fig.suptitle(f"Detected portion of randomly chosen subshowers\n{context}", y=0.995)
    fig.tight_layout()
    save(fig, path, written)


def plot_cut(
    responses,
    summaries,
    array,
    folder,
    context,
    primaries,
    seed,
    report,
    written,
    ground_options,
):
    """
    Plots for one energy cut, from the responses of its subshowers and, with
    ground_options, their ground summaries; returns the record of them.
    """
    record = observables.record_of(responses, array)
    if ground_options is not None:
        frame, _ = ground_options
        record.update(ground_observables.entries_of(summaries, frame, responses, array))
    sources = responses.values["source"]
    pdgs = responses.values["start_pdg"]
    by_primary = [primaries[source] for source in sources]
    plot_groups(
        observables.split_record(record, observables.fold_small_groups(by_primary)),
        folder,
        "by_primary",
        f"{context}, by primary energy",
        written,
        ground_options,
    )
    by_pdg = observables.fold_small_groups([f"pdg {pdg}" for pdg in pdgs])
    plot_groups(
        observables.split_record(record, by_pdg),
        folder,
        "by_pdg",
        f"{context}, by start particle",
        written,
        ground_options,
    )
    electrons = np.isin(pdgs, ELECTRONS_AND_POSITRONS)
    plot_groups(
        observables.split_record(
            record,
            observables.fold_small_groups(
                [g if keep else None for g, keep in zip(by_primary, electrons)]
            ),
        ),
        folder,
        "by_primary_electrons_and_positrons",
        f"{context}, electrons and positrons by primary energy",
        written,
        ground_options,
    )
    plot_examples(
        responses, array, os.path.join(folder, "examples.png"), context, seed, written
    )
    if not responses.detected.any() and len(responses):
        report(
            "  No subshower deposited energy. If that's unexpected, check that"
            " the array is where the subshowers reach the ground:"
        )
        for line in coordinate_ranges(responses, array):
            report(f"    {line}")
    return record


# -------------------------------------------------------------------- main


def main():
    args = parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    report = Report()
    try:
        array = detector.from_config(config)
        ground_settings = GroundSettings.from_config(config)
        showers_folder = generation.showers_folder(config)
        queued = generation.queued_outputs(showers_folder) or {}
        checks = datasets.check_subshower_sets(config, queued)
    except ground.GroundConfigError as error:
        sys.exit(f"Can't plot from {args.config_file}: ground: {error}")
    except (detector.DetectorConfigError, generation.ConfigError) as error:
        sys.exit(f"Can't plot from {args.config_file}: {error}")
    except KeyError as error:
        sys.exit(f"Can't plot from {args.config_file}: it has no {error} entry")

    run_name = (
        config.get("run_name")
        or os.path.splitext(os.path.basename(args.config_file))[0]
    )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(showers_folder, "plots", str(run_name))
    )
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as error:
        sys.exit(f"Can't make the output folder {output_dir}: {error}")
    unseeded = detector.is_unseeded(config)
    report(f"Config: {os.path.abspath(args.config_file)}")
    report(f"Detector: {describe_detector(config, array)}")
    report(f"Ground: {describe_ground(ground_settings, args.frame)}")
    report(f"Output: {output_dir}")
    if unseeded:
        report(
            "The detector array is random and has no seed, so it's different"
            " every run: responses are measured again each time, and can't be"
            " compared between runs. Give detectors: generator a seed to fix it."
        )
    if checks and not checks[0].subshower_set.leaves_only:
        report(
            "subshowers: leaves_only is false, but the sampler takes every"
            " particle in a subshower to be a leaf, so deposits, and particles"
            " on the ground, count particles that went on to make others."
        )
    report()

    ready = [check for check in checks if check.ready]
    missing = [check for check in checks if not check.ready]
    report(f"Subshower sets in {showers_folder}: {len(ready)} of {len(checks)} ready")
    report.items(
        f"Missing {len(missing)}:",
        [f"{check.label}: {check.problem}" for check in missing],
    )
    if missing:
        report(
            "  scripts/generate_data_from_config.py submits jobs to make"
            " what's missing."
        )

    measured = []  # (check, responses, ground summaries)
    if ready:
        report()
        report("Detector responses, and what reaches the ground:")
    for check in ready:
        source = None
        try:
            try:
                settings = responses_settings(check.subshower_set, config, args)
                responses, how, source = measure(
                    check,
                    array,
                    settings,
                    responses_path(output_dir, check.subshower_set),
                    args,
                    reuse=not (args.recompute or unseeded),
                )
            except Exception as error:  # one unreadable set shouldn't stop the rest
                report(
                    f"  {check.label}: couldn't be read,"
                    f" {type(error).__name__}: {error}"
                )
                continue
            n_in_set = responses.metadata.get("n_subshowers_in_set", len(responses))
            n_detected = int(responses.detected.sum())
            of = "" if len(responses) == n_in_set else f" chosen from {n_in_set}"
            report(
                f"  {check.label}: {len(responses)} subshowers{of},"
                f" {n_detected} ({percent(n_detected, len(responses))}) deposited"
                f" energy; {how}"
            )
            # nor should the ground stop the responses being plotted
            try:
                summaries, ground_how, source = summarise_ground(
                    check,
                    responses,
                    ground_settings_for(settings, ground_settings),
                    ground_settings,
                    ground_path(output_dir, check.subshower_set),
                    reuse=not args.recompute,
                    source=source,
                )
                n_on_ground = int(summaries.reaches_ground.sum())
                report(
                    f"    {n_on_ground} ({percent(n_on_ground, len(summaries))})"
                    f" put particles on the ground; {ground_how}"
                )
            except Exception as error:
                summaries = None
                report(
                    "    what reaches the ground couldn't be worked out,"
                    f" {type(error).__name__}: {error}"
                )
            measured.append((check, responses, summaries))
        finally:
            if source is not None:
                source.release()

    report_path = os.path.join(output_dir, "report.txt")
    if not measured:
        report()
        report("No subshower set could be used, so there is nothing to plot.")
        report.write(report_path)
        sys.exit(1)

    primaries = {
        check.subshower_set.shower.subfolder: primary_label(check.subshower_set.shower)
        for check, _, _ in measured
    }
    particle = os.path.basename(showers_folder)
    only = ", detected only" if args.detected_only else ""
    ground_options = (args.frame, ground_settings.reference_distance[args.frame])
    by_cut = {}
    for check, responses, summaries in measured:
        by_cut.setdefault(check.subshower_set.energy_cut, []).append(
            (responses, summaries)
        )
    written = []
    records = {}
    all_on_ground = True  # whether every cut plotted has its ground observables
    for cut, parts in by_cut.items():
        responses = Responses.concatenate(part[0] for part in parts)
        summaries = None
        if all(part[1] is not None for part in parts):
            summaries = GroundSummaries.concatenate(part[1] for part in parts)
        if args.detected_only:
            if summaries is not None:
                summaries = summaries.select(responses.detected)
            responses = responses.select(responses.detected)
        report()
        report(f"Energy cut {cut} GeV: plotting {len(responses)} subshowers")
        if not len(responses):
            continue
        if summaries is None:
            all_on_ground = False
            report(
                "  without what reaches the ground, as that couldn't be worked"
                " out for every set"
            )
        context = f"{particle}, energy cut {cut} GeV{only}"
        records[f"cut {cut} GeV"] = plot_cut(
            responses,
            summaries,
            array,
            os.path.join(output_dir, f"e{cut}"),
            context,
            primaries,
            args.seed,
            report,
            written,
            ground_options if summaries is not None else None,
        )
    if len(records) > 1:
        report()
        report(f"Comparing {len(records)} energy cuts")
        plot_groups(
            records,
            os.path.join(output_dir, "all_cuts"),
            "by_cut",
            f"{particle}, by energy cut{only}",
            written,
            ground_options if all_on_ground else None,
        )

    folders = {}
    for path in written:
        folder = os.path.relpath(os.path.dirname(path), output_dir)
        folders[folder] = folders.get(folder, 0) + 1
    listed = ", ".join(f"{folder}/ ({count})" for folder, count in folders.items())
    report()
    report(f"Wrote {len(written)} plots in {output_dir}: {listed}")
    report.write(report_path)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
