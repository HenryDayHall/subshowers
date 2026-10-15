import re as _re

import numpy as _np
from matplotlib import pyplot as _plt

# Group colours, given out in this order. As a set they keep neighbouring
# colours apart under colour-vision deficiency, and the first three are
# apart in every pair. Groups past the eighth, and OTHER, are grey: fold
# small groups into OTHER rather than plot more than about seven.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
GREY = "#8a8984"
OTHER = "other"


def _natural_key(name):
    """Sorts numbers within names by value, e.g. cut 900 before cut 1000."""
    parts = _re.split(r"(\d+)", str(name))
    return [int(part) if part.isdigit() else part for part in parts]


def group_colours(group_names):
    """
    {group name: colour}, in the order to draw and list the groups: sorted,
    with numbers in names compared by value, and OTHER last. Groups keep
    their colours whenever the same set of groups is plotted.
    """
    names = sorted((name for name in group_names if name != OTHER), key=_natural_key)
    colours = {
        name: PALETTE[i] if i < len(PALETTE) else GREY for i, name in enumerate(names)
    }
    if OTHER in group_names:
        colours[OTHER] = GREY
    return colours


def bin_edges(values, log=False, n_edges=40):
    """
    Edges spanning the finite values: geometric if log and the values are
    all positive, otherwise even. A single value gets a range around it, as
    histograms need increasing edges.
    """
    values = _np.asarray(values, dtype=float)
    values = values[_np.isfinite(values)]
    if values.size == 0:
        return _np.linspace(0.0, 1.0, n_edges)
    lo, hi = values.min(), values.max()
    if log and lo > 0:
        if lo == hi:
            lo, hi = lo / 2, hi * 2
        return _np.geomspace(lo, hi, n_edges)
    if lo == hi:
        half_width = abs(lo) * 0.1 if lo else 0.5
        lo, hi = lo - half_width, hi + half_width
    return _np.linspace(lo, hi, n_edges)


def scatter_with_marginals(
    gridspec,
    by_grouping,
    x_key,
    y_key,
    x_get,
    y_get,
    x_label,
    y_label,
    x_log=True,
    y_log=False,
    group_name_prefix=None,
    **kwargs,
):
    """
    Draw one scatter panel (start KE on x) with marginal histograms, one
    colour per group, into the sub-gridspec ``gridspec``.
    """
    inner = gridspec.subgridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.05,
        hspace=0.05,
    )
    fig = _plt.gcf()
    ax = fig.add_subplot(inner[1, 0])
    ax_topx = fig.add_subplot(inner[0, 0], sharex=ax)
    ax_righty = fig.add_subplot(inner[1, 1], sharey=ax)
    # which="both", as log axes spanning less than a decade label minor ticks
    ax_topx.tick_params(which="both", labelbottom=False)
    ax_righty.tick_params(which="both", labelleft=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    group_name_colours = group_colours(by_grouping)
    group_names = list(group_name_colours)

    all_x, all_y = [], []
    for group_name in group_names:
        record = by_grouping[group_name]
        xs = _np.asarray([x_get(v) for v in record[x_key]], dtype=float)
        ys = _np.asarray([y_get(v) for v in record[y_key]], dtype=float)
        mask = _np.isfinite(xs) & _np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        label = f"{group_name_prefix}{group_name}" if group_name_prefix else group_name
        ax.scatter(
            xs,
            ys,
            s=8,
            alpha=0.5,
            color=group_name_colours[group_name],
            label=label,
            **kwargs,
        )
        all_x.append(xs)
        all_y.append(ys)

    if not all_x:
        return ax
    all_x = _np.concatenate(all_x)
    all_y = _np.concatenate(all_y)

    x_bins = bin_edges(all_x, x_log)
    y_bins = bin_edges(all_y, y_log)
    for group_name in group_names:
        record = by_grouping[group_name]
        xs = _np.asarray([x_get(v) for v in record[x_key]], dtype=float)
        ys = _np.asarray([y_get(v) for v in record[y_key]], dtype=float)
        mask = _np.isfinite(xs) & _np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        ax_topx.hist(
            xs, bins=x_bins, histtype="step", color=group_name_colours[group_name]
        )
        ax_righty.hist(
            ys,
            bins=y_bins,
            histtype="step",
            color=group_name_colours[group_name],
            orientation="horizontal",
        )

    if x_log and all_x.min() > 0:
        ax.set_xscale("log")
    if y_log and all_y.min() > 0:
        ax.set_yscale("log")
    ax.legend(fontsize=7)
    return ax


def spectrum_histogram(
    ax,
    by_grouping,
    key,
    x_label,
    log_x=False,
    combined=False,
    group_name_prefix=None,
    **kwargs,
):
    """
    Overlay, one line per group, the average per-sample frequency of a
    per-sample spectrum (a list/array of values for each subshower).  The
    bin range spans every value seen across all samples and group_names.  A shaded
    band shows the 0.25 and 0.75 quantiles of the per-sample bin counts.

    With ``combined=True`` every group is pooled into a single line.
    """
    if combined:
        merged = {key: [s for store in by_grouping.values() for s in store[key]]}
        by_grouping = {"all": merged}
    group_name_colours = group_colours(by_grouping)
    group_names = list(group_name_colours)

    # per-group_name list of per-sample value arrays, and the pooled values
    per_sample = {}
    pooled = {}
    all_values = []
    for group_name in group_names:
        spectra = [
            _np.asarray(s, dtype=float).ravel() for s in by_grouping[group_name][key]
        ]
        per_sample[group_name] = spectra
        values = _np.concatenate(spectra) if spectra else _np.array([])
        pooled[group_name] = values
        if values.size:
            all_values.append(values)

    if not all_values:
        ax.set_xlabel(x_label)
        ax.set_ylabel("count per sample")
        return

    all_values = _np.concatenate(all_values)
    lo = all_values.min()
    bins = bin_edges(all_values, log_x)
    centres = 0.5 * (bins[:-1] + bins[1:])

    for group_name in group_names:
        spectra = per_sample[group_name]
        if not spectra:
            continue
        # bin counts for each sample: (n_samples, n_bins)
        per_sample_counts = _np.array(
            [_np.histogram(values, bins=bins)[0] for values in spectra], dtype=float
        )
        mean_counts = per_sample_counts.mean(axis=0)
        lower, upper = _np.quantile(per_sample_counts, [0.25, 0.75], axis=0)

        colour = group_name_colours[group_name]
        if group_name == "all":
            label = "all"
        elif group_name_prefix is not None:
            label = f"{group_name_prefix} {group_name}"
        else:
            label = group_name
        ax.step(centres, mean_counts, where="mid", color=colour, label=label, **kwargs)
        ax.fill_between(centres, lower, upper, step="mid", color=colour, alpha=0.2)

    if log_x and lo > 0:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("count per sample")
    ax.legend(fontsize=7)
    ax.semilogy()
