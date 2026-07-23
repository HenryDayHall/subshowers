import numpy as _np
from matplotlib import pyplot as _plt


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
    ax_topx.tick_params(labelbottom=False)
    ax_righty.tick_params(labelleft=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    group_names = sorted(by_grouping)
    colours = _plt.cm.tab10(_np.linspace(0, 1, len(group_names)))
    group_name_colours = dict(zip(group_names, colours))

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

    x_bins = (
        _np.geomspace(all_x.min(), all_x.max(), 40)
        if x_log
        else _np.linspace(all_x.min(), all_x.max(), 40)
    )
    y_bins = (
        _np.geomspace(all_y.min(), all_y.max(), 40)
        if y_log and all_y.min() > 0
        else _np.linspace(all_y.min(), all_y.max(), 40)
    )
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

    if x_log:
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
    group_names = sorted(by_grouping)
    colours = _plt.cm.tab10(_np.linspace(0, 1, len(group_names)))
    group_name_colours = dict(zip(group_names, colours))

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
    lo, hi = all_values.min(), all_values.max()
    if log_x and lo > 0:
        bins = _np.geomspace(lo, hi, 40)
    else:
        bins = _np.linspace(lo, hi, 40)
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
