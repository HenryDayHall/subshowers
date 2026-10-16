"""
Records and figures of the ground-level observables of subshowers (see
subshowers.ground): lateral distributions of particles and of the detector
signal, energy spectra, particle numbers and their fluctuations, arrival
times, and readings at a reference distance from the axis.

A record, as in subshowers.observables, has a list for each value, with an
entry per subshower. record_of gives one with the start values and the
ground observables; entries_of gives just the ground observables, to add to
a record of the same subshowers' detector responses (observables.record_of).

The figures take {group: record}, like those in observables, and draw a line
for each group: the mean over its subshowers, with a band of the standard
error of that mean. Given reference, the name of one group, the lateral
distributions and energy spectra also get panels of each group's ratio to
it, e.g. a generative model's to the dataset's.

Records can hold the observables of all the particles that reach the
ground, under keys starting ground_, and of only those that hit a detector,
under keys starting detected_ (see PARTICLES); each figure is drawn of one
or the other with particles="all" or "detected". The detected particles are
counted in the same bins and divided by the same areas (of the shower plane,
not of the detectors), so their curves are the part of the others that the
array observes. A generative model that only makes particles reaching the
detectors is compared with the dataset's detected particles; groups without
the observables a figure draws are listed, but have no lines.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from subshowers import ground, observables
from subshowers.population_plotting import GREY, bin_edges, group_colours

LINE_WIDTH = 1.5
MARKER_SIZE = 4
BAND_ALPHA = 0.2
GRID = {"color": "#e4e3df", "linewidth": 0.5}
MUTED = "#6b6a66"
N_ENERGY_BINS = 10  # bins of start kinetic energy, for profiles against it

# For each of ground.PARTICLES: the start of the keys of their observables
# in a record, and how figures describe them
PARTICLES = {
    "all": {"key": "ground", "where": "reaching the ground", "adjective": ""},
    "detected": {
        "key": "detected",
        "where": "reaching a detector",
        "adjective": "detected ",
    },
}


def _words(particles):
    if particles not in PARTICLES:
        raise ValueError(
            f"particles must be one of {list(PARTICLES)}, not {particles!r}"
        )
    return PARTICLES[particles]


# ------------------------------------------------------------------ records


def entries_of(summaries, frame="subshower", responses=None, array=None):
    """
    {name: list with an entry per subshower} of the ground observables of a
    ground.GroundSummaries, measuring distances and delays about the axis of
    frame ("subshower" or "shower"). Their names start ground_, or for
    summaries of detected particles, detected_. Given the
    responses.Responses of the same subshowers and the array they were
    measured with, the detector signal about the same axis is added.
    """
    if frame not in ground.FRAMES:
        raise ValueError(f"frame must be one of {ground.FRAMES}, not {frame!r}")
    f = ground.FRAMES.index(frame)
    key = _words(summaries.particles)["key"]
    histograms = summaries.histograms
    entries = {
        f"{key}_number": list(histograms["number"]),
        f"{key}_spectrum": list(histograms["spectrum"]),
        f"{key}_lateral": list(histograms["lateral"][:, f]),
        f"{key}_delay_sum": list(histograms["delay_sum"][:, f]),
        f"{key}_delay_sum2": list(histograms["delay_sum2"][:, f]),
        f"{key}_at_reference": list(histograms["at_reference"][:, f]),
    }
    if responses is not None:
        if array is None:
            raise ValueError(
                "the signal needs the array the responses were measured with"
            )
        ground.check_aligned(summaries, responses)
        points, directions = summaries.axes(frame)
        deposit, area, at_reference = ground.signal_about_axes(
            responses, array, points, directions, summaries.settings, frame
        )
        entries["signal_deposit"] = list(deposit)
        entries["signal_area"] = list(area)
        entries["signal_at_reference"] = list(at_reference)
    return entries


def record_of(summaries, frame="subshower", responses=None, array=None):
    """
    The record of every subshower in a ground.GroundSummaries: its start
    values, as observables.record_of gives them, and entries_of.
    """
    record = {
        name: list(summaries.values[name]) for name in observables.conditioning_values
    }
    record.update(entries_of(summaries, frame, responses, array))
    return record


# ------------------------------------------------------------------ helpers


def _values(record, key):
    """A record's entries for key as one array, (n subshowers, ...), or None."""
    values = record.get(key)
    if values is None or len(values) == 0:
        return None
    return np.stack([np.asarray(value, dtype=float) for value in values])


def _stacked(by_group, key):
    """{group: _values of its record}, for the groups whose records have key."""
    stacked = {group: _values(record, key) for group, record in by_group.items()}
    return {group: values for group, values in stacked.items() if values is not None}


def _mean_and_error(values):
    """
    The mean over the first axis (the subshowers), leaving out nan, the
    standard error of that mean, and the number of values it's the mean of.
    """
    values = np.asarray(values, dtype=float)
    known = np.isfinite(values)
    count = known.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(known, values, 0.0).sum(axis=0) / count
        squares = np.where(known, (values - mean) ** 2, 0.0).sum(axis=0)
        error = np.sqrt(squares / (count - 1) / count)
    mean = np.where(count > 0, mean, np.nan)
    error = np.where(count > 1, error, np.nan)
    return mean, error, count


def _centres(edges):
    """
    Geometric centres of logarithmic bins. A first bin starting at 0 (the
    disc of ground.RADIUS_EDGES) is put at half its outer edge.
    """
    edges = np.asarray(edges, dtype=float)
    lower = np.where(edges[:-1] > 0, edges[:-1], edges[1:] / 4)
    return np.sqrt(lower * edges[1:])


def _ring_areas(edges):
    return np.pi * np.diff(np.asarray(edges, dtype=float) ** 2)


def _axis_label(frame):
    return f"distance from the {frame} axis [m]"


def _no_values(ax, text="no values"):
    ax.text(
        0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes
    )


def _style(ax):
    ax.grid(True, which="major", **GRID)
    ax.set_axisbelow(True)


def _panel(fig, cell, ratio):
    """A main panel, and a ratio panel under it if ratio, sharing its x axis."""
    if not ratio:
        ax = fig.add_subplot(cell)
        _style(ax)
        return ax, None
    inner = cell.subgridspec(2, 1, height_ratios=(3, 1), hspace=0.05)
    ax = fig.add_subplot(inner[0])
    ratio_ax = fig.add_subplot(inner[1], sharex=ax)
    ax.tick_params(which="both", labelbottom=False)
    _style(ax)
    _style(ratio_ax)
    return ax, ratio_ax


def _log_limits(values, pad=3.0):
    """Limits for a log axis spanning the positive finite values, or None."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return None
    return values.min() / pad, values.max() * pad


def _one_two_five(value, _=None):
    """A tick label for 1, 2 and 5 times a power of ten; none for others."""
    if not value > 0:
        return ""
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = value / 10.0**exponent
    if not np.isclose(mantissa, [1, 2, 5], rtol=1e-6).any():
        return ""
    if 1e-2 <= value < 1e4:
        return f"{value:g}"
    if np.isclose(mantissa, 1):
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:.0f}\\times10^{{{exponent}}}$"


def _tidy_log_ticks(ax):
    """
    Label log axes clearly: with two or more powers of ten in range, only
    those; with fewer, 1, 2 and 5 times powers of ten, so there's always a
    label or two.
    """
    for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
        if axis.get_scale() != "log":
            continue
        lo, hi = sorted(limits)
        if not lo > 0:
            continue
        powers = np.floor(np.log10(hi) + 1e-9) - np.ceil(np.log10(lo) - 1e-9) + 1
        if powers >= 2:
            axis.set_minor_formatter(NullFormatter())
        else:
            axis.set_major_formatter(FuncFormatter(_one_two_five))
            axis.set_minor_formatter(FuncFormatter(_one_two_five))


def _draw_curves(ax, x, curves, colours, log_x=True, log_y=True):
    """
    A line for each group's mean (a marker on each bin, so lone bins show),
    with a band of its standard error. curves is {group: (mean, error)}.
    Returns {group: line} for the groups drawn.
    """
    handles, shown_x, shown_y = {}, [], []
    for group, (mean, error) in curves.items():
        mean = np.asarray(mean, dtype=float)
        shown = np.isfinite(mean) & ((mean > 0) if log_y else True)
        if not shown.any():
            continue
        y = np.where(shown, mean, np.nan)
        (line,) = ax.plot(
            x,
            y,
            color=colours[group],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        error = np.asarray(error, dtype=float)
        banded = shown & np.isfinite(error)
        if banded.any():
            lower = y - error
            if log_y:
                lower = np.where(lower > 0, lower, y * 1e-3)
            ax.fill_between(
                x,
                lower,
                y + error,
                where=banded,
                color=colours[group],
                alpha=BAND_ALPHA,
                linewidth=0,
            )
        handles[group] = line
        shown_x.append(np.asarray(x)[shown])
        shown_y.append(y[shown])
    if not handles:
        return handles
    shown_x, shown_y = np.concatenate(shown_x), np.concatenate(shown_y)
    if log_x:
        ax.set_xscale("log")
        limits = _log_limits(shown_x, pad=1.6)
        if limits is not None:
            ax.set_xlim(*limits)
    if log_y:
        ax.set_yscale("log")
        limits = _log_limits(shown_y)
        if limits is not None:
            ax.set_ylim(*limits)
    _tidy_log_ticks(ax)
    return handles


def _draw_ratios(ax, x, curves, colours, reference):
    """Each group's mean over the reference group's, with its error band."""
    reference_mean, reference_error = (np.asarray(v, float) for v in curves[reference])
    ratios = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for group, (mean, error) in curves.items():
            colour = colours[group]
            if group == reference:
                band = reference_error / reference_mean
                ax.axhline(1.0, color=colour, linewidth=1.0)
                known = np.isfinite(band) & (reference_mean > 0)
                if known.any():
                    ax.fill_between(
                        x, 1 - band, 1 + band, where=known, color=colour,
                        alpha=BAND_ALPHA, linewidth=0,
                    )  # fmt: skip
                continue
            ratio = np.asarray(mean, float) / reference_mean
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            ratio_error = np.asarray(error, float) / reference_mean
            ax.plot(
                x, ratio, color=colour, linewidth=LINE_WIDTH, marker="o",
                markersize=MARKER_SIZE - 1, markeredgecolor="white",
                markeredgewidth=0.5,
            )  # fmt: skip
            known = np.isfinite(ratio) & np.isfinite(ratio_error)
            if known.any():
                ax.fill_between(
                    x, ratio - ratio_error, ratio + ratio_error, where=known,
                    color=colour, alpha=BAND_ALPHA, linewidth=0,
                )  # fmt: skip
            ratios.append(ratio[np.isfinite(ratio)])
    shown = np.concatenate(ratios) if ratios else np.array([])
    if len(shown):
        lo = max(0.0, min(shown.min(), 0.8) - 0.1)
        hi = min(3.0, max(shown.max(), 1.2) + 0.1)
        ax.set_ylim(lo, hi)
    else:
        ax.set_ylim(0.5, 1.5)
    ax.set_ylabel(f"ratio to\n{reference}", fontsize=8)


def _mark_reference(ax, distance):
    """A thin line at the reference distance, named at its top."""
    if distance is None:
        return
    ax.axvline(distance, color=GREY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    ax.text(
        distance, 0.98, f" {distance:g} m", transform=ax.get_xaxis_transform(),
        ha="left", va="top", fontsize=7, color=MUTED, clip_on=True,
    )  # fmt: skip


def _legend(fig, by_group, title, group_name_prefix=None):
    """
    The figure's title, and under the panels a legend of every group, in
    the order of its colours, whether or not anything of it was drawn.
    """
    fig.suptitle(title)
    colours = group_colours(by_group)
    if not colours:
        return
    handles, labels = [], []
    for group, colour in colours.items():
        handles.append(
            Line2D(
                [], [], color=colour, linewidth=LINE_WIDTH, marker="o",
                markersize=MARKER_SIZE, markeredgecolor="white", markeredgewidth=0.6,
            )  # fmt: skip
        )
        name = f"{group_name_prefix}{group}" if group_name_prefix else str(group)
        n_subshowers = observables.record_length(by_group[group])
        labels.append(f"{name} ({n_subshowers} subshowers)")
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(handles), 4),
        frameon=False,
    )


def _check_reference(by_group, reference):
    if reference is not None and reference not in by_group:
        raise ValueError(
            f"reference {reference!r} isn't one of the groups {list(by_group)}"
        )


def _energy_edges(by_group):
    """Bins of start kinetic energy shared by all the groups."""
    energies = np.concatenate(
        [
            np.asarray(record["start_kinetic_energy"], float)
            for record in by_group.values()
        ]
    )
    return bin_edges(energies, log=True, n_edges=N_ENERGY_BINS + 1)


def _by_energy(record, values, edges, statistic="mean"):
    """
    For each bin of start kinetic energy, a statistic of values over the
    subshowers in it, and its standard error: the mean, or the relative
    spread (standard deviation over mean).
    """
    energy = np.asarray(record["start_kinetic_energy"], dtype=float)
    which = np.clip(np.searchsorted(edges, energy, side="right") - 1, 0, len(edges) - 2)
    centre, error = [], []
    for b in range(len(edges) - 1):
        in_bin = values[which == b]
        in_bin = in_bin[np.isfinite(in_bin)]
        n = len(in_bin)
        with np.errstate(invalid="ignore", divide="ignore"):
            if statistic == "mean":
                centre.append(in_bin.mean() if n else np.nan)
                error.append(in_bin.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
            else:
                mean = in_bin.mean() if n else np.nan
                spread = in_bin.std(ddof=1) / mean if n > 1 and mean > 0 else np.nan
                centre.append(spread)
                error.append(
                    spread * np.sqrt(1 / (2 * (n - 1)) + spread**2 / n)
                    if n > 2
                    else np.nan
                )
    return np.array(centre), np.array(error)


# ------------------------------------------------------------------ figures


def lateral_distributions(
    by_group,
    frame="subshower",
    reference=None,
    reference_distance=None,
    title=None,
    particles="all",
):
    """
    The density of each kind of particle on the ground against distance
    from the axis (the lateral distribution function), as particles per m²
    of the shower plane, and, if the records have it, the energy deposited
    per m² of detector against the detector's distance from the axis. With
    particles="detected", only the particles that hit a detector are
    counted, still per m² of the shower plane.
    """
    words = _words(particles)
    _check_reference(by_group, reference)
    colours = group_colours(by_group)
    has_signal = any("signal_deposit" in record for record in by_group.values())
    columns = list(ground.SPECIES) + (["signal"] if has_signal else [])
    fig = plt.figure(
        figsize=(3.9 * len(columns), 5.4 if reference else 4.3), layout="constrained"
    )
    grid = fig.add_gridspec(1, len(columns))
    x = _centres(ground.RADIUS_EDGES)
    areas = _ring_areas(ground.RADIUS_EDGES)
    lateral = _stacked(by_group, f"{words['key']}_lateral")
    deposits = _stacked(by_group, "signal_deposit")
    detector_areas = _stacked(by_group, "signal_area")
    for c, column in enumerate(columns):
        ax, ratio_ax = _panel(fig, grid[0, c], reference is not None)
        curves = {}
        for group in colours:
            if column == "signal":
                if group not in deposits or group not in detector_areas:
                    continue
                deposit, area = deposits[group], detector_areas[group]
                with np.errstate(invalid="ignore", divide="ignore"):
                    density = np.where(area > 0, deposit / area, np.nan)
            else:
                if group not in lateral:
                    continue
                density = lateral[group][:, ground.SPECIES.index(column)] / areas
            curves[group] = _mean_and_error(density)[:2]
        drawn = _draw_curves(ax, x, curves, colours)
        if column == "signal":
            ax.set_title("detector signal")
            ax.set_ylabel("deposited energy per m² of detector [GeV m⁻²]")
        else:
            ax.set_title(ground.SPECIES_LABELS[column])
            if c == 0:
                ax.set_ylabel(
                    f"{words['adjective']}particles per m² (mean per subshower)"
                )
        if not drawn:
            _no_values(ax)
            if ratio_ax is not None:
                ratio_ax.set_ylim(0.5, 1.5)
        else:
            _mark_reference(ax, reference_distance)
            if ratio_ax is not None and reference in drawn:
                _draw_ratios(ratio_ax, x, curves, colours, reference)
        (ratio_ax or ax).set_xlabel(_axis_label(frame))
    _legend(
        fig, by_group, title or f"Lateral distributions of particles {words['where']}"
    )
    return fig


def energy_spectra(by_group, reference=None, title=None, particles="all"):
    """
    The kinetic energy spectrum of each kind of particle reaching the
    ground, or with particles="detected" a detector: particles per decade of
    energy, as a mean per subshower. Energies are as the particles were
    made (see subshowers.ground).
    """
    words = _words(particles)
    _check_reference(by_group, reference)
    colours = group_colours(by_group)
    columns = list(ground.SPECIES)
    fig = plt.figure(
        figsize=(3.9 * len(columns), 5.4 if reference else 4.3), layout="constrained"
    )
    grid = fig.add_gridspec(1, len(columns))
    edges = ground.ENERGY_EDGES
    x = _centres(edges)
    widths = np.diff(np.log10(edges))
    spectra = _stacked(by_group, f"{words['key']}_spectrum")
    for c, species in enumerate(columns):
        ax, ratio_ax = _panel(fig, grid[0, c], reference is not None)
        curves = {}
        for group, spectrum in spectra.items():
            per_decade = spectrum[:, ground.SPECIES.index(species)] / widths
            curves[group] = _mean_and_error(per_decade)[:2]
        drawn = _draw_curves(ax, x, curves, colours)
        ax.set_title(ground.SPECIES_LABELS[species])
        if c == 0:
            ax.set_ylabel(
                f"{words['adjective']}particles per decade of energy\n"
                "(mean per subshower)"
            )
        if not drawn:
            _no_values(ax)
            if ratio_ax is not None:
                ratio_ax.set_ylim(0.5, 1.5)
        elif ratio_ax is not None and reference in drawn:
            _draw_ratios(ratio_ax, x, curves, colours, reference)
        (ratio_ax or ax).set_xlabel("kinetic energy [GeV]")
    _legend(fig, by_group, title or f"Energy spectra of particles {words['where']}")
    return fig


def particle_numbers(by_group, title=None, particles="all"):
    """
    How many of each kind of particle reach the ground, or with
    particles="detected" a detector: the fraction of subshowers with at
    least N, the mean number against start kinetic energy, and the relative
    fluctuations (standard deviation over mean) against start kinetic
    energy.
    """
    words = _words(particles)
    colours = group_colours(by_group)
    columns = list(ground.SPECIES)
    fig, axarr = plt.subplots(
        3, len(columns), figsize=(3.9 * len(columns), 10.5), layout="constrained"
    )
    edges = _energy_edges(by_group)
    centres = np.sqrt(edges[:-1] * edges[1:])
    all_numbers = _stacked(by_group, f"{words['key']}_number")
    for c, species in enumerate(columns):
        s = ground.SPECIES.index(species)
        label = ground.SPECIES_LABELS[species]
        some = f"{words['adjective']}{label}"
        numbers = {group: values[:, s] for group, values in all_numbers.items()}

        ax = axarr[0, c]
        _style(ax)
        drawn = False
        for group, values in numbers.items():
            values = values[np.isfinite(values)]
            positive = np.sort(values[values >= 1])
            if not len(positive):
                continue
            unique, first = np.unique(positive, return_index=True)
            at_least = (len(positive) - first) / len(values)
            ax.step(
                unique,
                at_least,
                where="post",
                color=colours[group],
                linewidth=LINE_WIDTH,
            )
            drawn = True
        ax.set_title(label)
        ax.set_xlabel(f"N, number of {label} {words['where']}")
        if c == 0:
            ax.set_ylabel("fraction of subshowers\nwith at least N")
        if drawn:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _tidy_log_ticks(ax)
        else:
            _no_values(ax, f"no subshower had\n{label} {words['where']}")

        for row, statistic, y_label in [
            (1, "mean", f"mean number of {some}"),
            (2, "spread", f"standard deviation / mean\nof the number of {some}"),
        ]:
            ax = axarr[row, c]
            _style(ax)
            curves = {
                group: _by_energy(by_group[group], values, edges, statistic)
                for group, values in numbers.items()
            }
            drawn = _draw_curves(
                ax, centres, curves, colours, log_y=(statistic == "mean")
            )
            ax.set_xlabel("start kinetic energy [GeV]")
            if c == 0 or statistic == "spread":
                ax.set_ylabel(y_label)
            if not drawn:
                _no_values(ax)
            elif statistic == "spread":
                ax.set_ylim(bottom=0)
    _legend(fig, by_group, title or f"Numbers of particles {words['where']}")
    return fig


def arrival_times(by_group, frame="subshower", title=None, particles="all"):
    """
    When particles reach the ground, against distance from the axis: their
    mean delay after a plane front moving along the axis at the speed of
    light, and the standard deviation of their delays (the thickness of the
    shower front). Both are over all the particles of a group's subshowers
    at that distance, or with particles="detected" those that hit a
    detector; the band is the standard error of the mean delay.
    """
    words = _words(particles)
    colours = group_colours(by_group)
    columns = list(ground.SPECIES)
    fig, axarr = plt.subplots(
        2, len(columns), figsize=(3.9 * len(columns), 7.5), layout="constrained"
    )
    x = _centres(ground.RADIUS_EDGES)
    counts = _stacked(by_group, f"{words['key']}_lateral")
    totals = _stacked(by_group, f"{words['key']}_delay_sum")
    all_squares = _stacked(by_group, f"{words['key']}_delay_sum2")
    for c, species in enumerate(columns):
        s = ground.SPECIES.index(species)
        label = ground.SPECIES_LABELS[species]
        means, thicknesses = {}, {}
        any_particles = False
        for group in colours:
            if not (group in counts and group in totals and group in all_squares):
                continue
            count = counts[group][:, s]
            total = totals[group][:, s]
            squares = all_squares[group][:, s]
            any_particles |= bool(np.nansum(count) > 0)
            known = np.isfinite(count) & np.isfinite(total) & np.isfinite(squares)
            n = np.where(known, count, 0).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(known, total, 0).sum(axis=0) / n
                variance = np.where(known, squares, 0).sum(axis=0) / n - mean**2
                thickness = np.sqrt(np.clip(variance, 0, None))
                thickness = np.where(n > 1, thickness, np.nan)
                error = thickness / np.sqrt(n)
            means[group] = (np.where(n > 0, mean, np.nan), error)
            thicknesses[group] = (thickness, np.full(len(x), np.nan))
        for row, curves, y_label in [
            (0, means, "mean arrival delay [ns]"),
            (1, thicknesses, "standard deviation of delays [ns]"),
        ]:
            ax = axarr[row, c]
            _style(ax)
            # delays span nanoseconds to microseconds; bins at no delay
            # (photons along the axis) aren't shown
            drawn = _draw_curves(ax, x, curves, colours)
            ax.set_xlabel(_axis_label(frame))
            if c == 0:
                ax.set_ylabel(y_label)
            if row == 0:
                ax.set_title(label)
            if not drawn:
                any_times = any(np.isfinite(m).any() for m, _ in means.values())
                if not any_particles:
                    _no_values(ax, f"no {label} {words['where']}")
                elif not any_times:
                    _no_values(ax, "no arrival times known")
                else:
                    _no_values(ax, "too few particles")
    _legend(fig, by_group, title or f"Arrival times of particles {words['where']}")
    return fig


def readings_at_reference(
    by_group, frame="subshower", reference_distance=None, title=None, particles="all"
):
    """
    What a ground array reads at the reference distance from the axis, like
    the Telescope Array's S800: the energy deposited per m² of detector
    there (S) for each subshower, and against start kinetic energy; and the
    density of electrons and positrons, and of muons, there against start
    kinetic energy, counting with particles="detected" only those that hit
    a detector. Subshowers with no detector at that distance are left out
    of S.
    """
    words = _words(particles)
    colours = group_colours(by_group)
    fig, axarr = plt.subplots(2, 2, figsize=(11.5, 9.5), layout="constrained")
    where = (
        f"{reference_distance:g} m from the {frame} axis"
        if reference_distance is not None
        else f"the reference distance from the {frame} axis"
    )
    edges = _energy_edges(by_group)
    centres = np.sqrt(edges[:-1] * edges[1:])

    signals = _stacked(by_group, "signal_at_reference")
    at_reference = _stacked(by_group, f"{words['key']}_at_reference")
    ax = axarr[0, 0]
    _style(ax)
    ax.set_title(f"S, at {where}")
    ax.set_xlabel("S, energy deposited per m² of detector [GeV m⁻²]")
    ax.set_ylabel("fraction of subshowers with a detector there")
    positive = [v[np.isfinite(v) & (v > 0)] for v in signals.values()]
    positive = np.concatenate(positive) if positive else np.array([])
    if not signals:
        _no_values(ax, "no detector responses")
    elif not len(positive):
        _no_values(ax, "no subshower deposited energy there")
    else:
        bins = bin_edges(positive, log=True, n_edges=25)
        labels, patches = [], []
        for group, values in signals.items():
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            nothing = np.mean(values == 0)
            shown = values[values > 0]
            *_, patch = ax.hist(
                shown, bins=bins, weights=np.full(len(shown), 1 / len(values)),
                histtype="step", linewidth=LINE_WIDTH, color=colours[group],
            )  # fmt: skip
            patches.append(patch[0])
            labels.append(f"{group}: {nothing:.0%} read nothing")
        ax.set_xscale("log")
        _tidy_log_ticks(ax)
        ax.legend(patches, labels, fontsize=7, frameon=False)

    panels = [
        (axarr[0, 1], "S", f"mean S at {where}\n[GeV m⁻²]"),
        (
            axarr[1, 0],
            "electrons",
            f"mean density of {words['adjective']}e± at\n{where} [m⁻²]",
        ),
        (
            axarr[1, 1],
            "muons",
            f"mean density of {words['adjective']}μ± at\n{where} [m⁻²]",
        ),
    ]
    for ax, quantity, y_label in panels:
        _style(ax)
        curves = {}
        for group in colours:
            record = by_group[group]
            if quantity == "S":
                values = signals.get(group)
            else:
                if group not in at_reference or reference_distance is None:
                    continue
                inner, outer = ground.annulus(reference_distance)
                area = np.pi * (outer**2 - inner**2)
                values = at_reference[group][:, ground.SPECIES.index(quantity)] / area
            if values is not None:
                curves[group] = _by_energy(record, values, edges)
        drawn = _draw_curves(ax, centres, curves, colours)
        ax.set_xlabel("start kinetic energy [GeV]")
        ax.set_ylabel(y_label)
        if not drawn:
            if quantity == "S" and not signals:
                _no_values(ax, "no detector responses")
            else:
                _no_values(ax, "nothing there")
    _legend(
        fig,
        by_group,
        title or f"Readings at the reference distance, of particles {words['where']}",
    )
    return fig
