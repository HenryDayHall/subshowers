from collections import Counter as _Counter, namedtuple as _namedtuple

import numpy as _np
from matplotlib import pyplot as _plt
from subshowers.population_plotting import (
    OTHER,
    bin_edges,
    group_colours,
    scatter_with_marginals,
    spectrum_histogram,
)

conditioning_values = [
    "start_kinetic_energy",
    "start_position",
    "start_direction",
    "start_time",
    "start_pdg",
]

output_values = [
    "total_deposit_energy",
    "max_deposit_energy",
    "mean_deposit_energy",
    "deposity_multiplicity",
    "deposit_energy_spectrum",
    "deposit_radial_spectrum",
    "mean_deposit_offset",
    "max_deposit_radius",
    "mean_deposit_radius",
    "deposit_sphericity",
    "deposit_energy_variance",
]


def get_outputs(sample, detector_array, output_dict=None):
    if output_dict is None:
        output_dict = {name: [] for name in output_values}
    # basics
    output_dict["total_deposit_energy"].append(sample.total_detected_energy)
    nonzero_idxs = sample.energy_per_detector > 0
    nonzero_deposits = sample.energy_per_detector[nonzero_idxs]
    output_dict["deposity_multiplicity"].append(len(nonzero_deposits))
    output_dict["deposit_energy_spectrum"].append(list(nonzero_deposits))
    if len(nonzero_deposits):
        output_dict["max_deposit_energy"].append(float(nonzero_deposits.max()))
        output_dict["mean_deposit_energy"].append(float(nonzero_deposits.mean()))
    else:
        output_dict["max_deposit_energy"].append(0.0)
        output_dict["mean_deposit_energy"].append(0.0)
    start_position = sample.start_position

    # positions (x, y) of the detectors that recorded energy
    hit_positions = detector_array.centers[nonzero_idxs][:, :2]
    total_energy = nonzero_deposits.sum()

    if len(nonzero_deposits) == 0 or total_energy == 0:
        output_dict["deposit_radial_spectrum"].append([])
        output_dict["mean_deposit_offset"].append(_np.nan)
        output_dict["max_deposit_radius"].append(0.0)
        output_dict["mean_deposit_radius"].append(0.0)
        output_dict["deposit_sphericity"].append(_np.nan)
        output_dict["deposit_energy_variance"].append(0.0)
        return output_dict

    # energy-weighted centroid of the deposits
    weights = nonzero_deposits / total_energy
    centroid = _np.sum(hit_positions * weights[:, None], axis=0)

    # offset of the deposit centroid from the subshower start (x, y)
    mean_deposit_offset = _np.linalg.norm(centroid - start_position[:2])
    output_dict["mean_deposit_offset"].append(mean_deposit_offset)

    # radial spread of the deposits about their centroid
    relative = hit_positions - centroid
    radii = _np.linalg.norm(relative, axis=-1)

    # deposited-energy radial profile about the centroid
    output_dict["deposit_radial_spectrum"].append(radii)

    output_dict["max_deposit_radius"].append(float(radii.max()))
    output_dict["mean_deposit_radius"].append(float(_np.sum(radii * weights)))

    # sphericity from the energy-weighted 2D spatial covariance:
    # ratio of smaller to larger eigenvalue (1 = round, 0 = linear)
    covariance = (relative * weights[:, None]).T @ relative
    eigenvalues = _np.linalg.eigvalsh(covariance)
    eigenvalues = _np.clip(eigenvalues, 0.0, None)
    if eigenvalues.max() > 0:
        sphericity = float(eigenvalues.min() / eigenvalues.max())
    else:
        sphericity = _np.nan
    output_dict["deposit_sphericity"].append(sphericity)

    # variance of the individual detector deposits
    output_dict["deposit_energy_variance"].append(float(_np.var(nonzero_deposits)))

    return output_dict


def new_record():
    """An empty record: {name: []} for each conditioning value and output."""
    return {name: [] for name in conditioning_values + output_values}


def add_sample(record, sample, detector_array):
    """Add one sample's conditioning values and outputs to a record."""
    for name in conditioning_values:
        record[name].append(getattr(sample, name))
    get_outputs(sample, detector_array, record)
    return record


def get_samples_per_source(sampler, array):
    per_source = {}
    n_total, n_detected = 0, 0
    for sample in sampler.iter_epoch(shuffle=True):
        n_total += 1
        n_detected += sample.n_hit_detectors > 0
        record = per_source.setdefault(sample.source.name, new_record())
        add_sample(record, sample, array)

    print(f"Epoch complete: {n_detected}/{n_total} subshowers deposited energy")
    return per_source


def record_of(responses, detector_array):
    """
    The record of every subshower in a responses.Responses, in order. The
    array must be the one the responses were measured with.
    """
    if len(detector_array) != responses.n_detectors:
        raise ValueError(
            f"the responses were measured with {responses.n_detectors} detectors,"
            f" not {len(detector_array)}"
        )
    record = new_record()
    for sample in responses.samples():
        add_sample(record, sample, detector_array)
    return record


def split_record(record, groups):
    """
    {group: record} from one record, given each subshower's group in order.
    Subshowers whose group is None are left out.
    """
    indices = {}
    for i, group in enumerate(groups):
        if group is not None:
            indices.setdefault(group, []).append(i)
    return {
        group: {key: [values[i] for i in chosen] for key, values in record.items()}
        for group, chosen in indices.items()
    }


def pool(records):
    """One record holding the subshowers of several."""
    pooled = new_record()
    for record in records:
        for key in pooled:
            pooled[key].extend(record[key])
    return pooled


def record_length(record):
    return len(record[conditioning_values[0]])


def fold_small_groups(groups, max_groups=8, other=OTHER):
    """
    Each subshower's group, with all but the max_groups - 1 largest groups
    put together as `other`, so that no more groups are plotted than can be
    told apart. None (left out) stays None.
    """
    groups = list(groups)
    counts = _Counter(group for group in groups if group is not None)
    if len(counts) <= max_groups:
        return groups
    kept = {group for group, _ in counts.most_common(max_groups - 1)}
    return [group if group is None or group in kept else other for group in groups]


def gather_by_pdg(per_source, keys=None):
    """
    Return {pdg: {key: np.ndarray of values}} pooling all sources.
    """
    if keys is None:
        keys = conditioning_values + output_values

    by_pdg = {}
    for record in per_source.values():
        pdgs = record["start_pdg"]
        for i, pdg in enumerate(pdgs):
            store = by_pdg.setdefault(pdg, {key: [] for key in keys})
            for key in keys:
                store[key].append(record[key][i])
    for pdg, store in by_pdg.items():
        for key in store.keys():
            try:
                store[key] = _np.asarray(store[key])
            except ValueError:
                pass
    return by_pdg


def filter_pdgs(per_source, keep_pdgs):
    """
    Return a new per_source dict retaining only subshowers whose
    ``start_pdg`` is in ``keep_pdgs``.
    """
    keep_pdgs = set(keep_pdgs)
    filtered = {}
    for source_name, record in per_source.items():
        keep_mask = [pdg in keep_pdgs for pdg in record["start_pdg"]]
        filtered[source_name] = {
            key: [value for value, keep in zip(values, keep_mask) if keep]
            for key, values in record.items()
        }
    return filtered


def observables_A(by_group, title=None, **kwargs):
    fig = _plt.figure(figsize=(14, 12))
    outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)

    scatter_with_marginals(
        outer[0, 0],
        by_group,
        "start_kinetic_energy",
        "start_time",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "start time",
        **kwargs,
    )
    scatter_with_marginals(
        outer[0, 1],
        by_group,
        "start_kinetic_energy",
        "start_position",
        lambda v: v,
        lambda v: v[2],
        "start kinetic energy [GeV]",
        "start height (z) [m]",
        **kwargs,
    )
    scatter_with_marginals(
        outer[1, 0],
        by_group,
        "start_kinetic_energy",
        "total_deposit_energy",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "total deposited energy [GeV]",
        y_log=True,
        **kwargs,
    )
    scatter_with_marginals(
        outer[1, 1],
        by_group,
        "start_kinetic_energy",
        "deposity_multiplicity",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "detector multiplicity",
        **kwargs,
    )

    fig.suptitle(title or "Subshower start conditions vs detected response", y=0.995)
    return fig


def observables_B(by_group, title=None, **kwargs):
    fig = _plt.figure(figsize=(14, 12))
    outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)

    scatter_with_marginals(
        outer[0, 0],
        by_group,
        "start_kinetic_energy",
        "mean_deposit_radius",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "mean deposit radius [m]",
        **kwargs,
    )
    scatter_with_marginals(
        outer[0, 1],
        by_group,
        "start_kinetic_energy",
        "max_deposit_radius",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "max deposit radius [m]",
        **kwargs,
    )
    scatter_with_marginals(
        outer[1, 0],
        by_group,
        "start_kinetic_energy",
        "deposit_sphericity",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "deposit sphericity",
        **kwargs,
    )
    scatter_with_marginals(
        outer[1, 1],
        by_group,
        "start_kinetic_energy",
        "mean_deposit_offset",
        lambda v: v,
        lambda v: v,
        "start kinetic energy [GeV]",
        "mean deposit offset [m]",
        **kwargs,
    )

    fig.suptitle(title or "Subshower start energy vs deposit shape", y=0.995)
    return fig


def observables_C(by_group, title=None, **kwargs):
    fig, axarr = _plt.subplots(2, 2, figsize=(14, 11))
    (ax_energy, ax_radial), (ax_energy_all, ax_radial_all) = axarr

    spectrum_histogram(
        ax_energy,
        by_group,
        "deposit_energy_spectrum",
        "deposit energy [GeV]",
        log_x=True,
        **kwargs,
    )
    ax_energy.set_title("Deposit energy spectrum (per pdg)")
    spectrum_histogram(
        ax_radial, by_group, "deposit_radial_spectrum", "deposit radius [m]", **kwargs
    )
    ax_radial.set_title("Deposit radial spectrum (per pdg)")

    spectrum_histogram(
        ax_energy_all,
        by_group,
        "deposit_energy_spectrum",
        "deposit energy [GeV]",
        log_x=True,
        combined=True,
        **kwargs,
    )
    ax_energy_all.set_title("Deposit energy spectrum (all pdg)")
    spectrum_histogram(
        ax_radial_all,
        by_group,
        "deposit_radial_spectrum",
        "deposit radius bin [m]",
        combined=True,
        **kwargs,
    )
    ax_radial_all.set_title("Deposit radial spectrum (all pdg)")

    fig.suptitle(title or "Average per-sample deposit spectra", y=0.995)
    _plt.tight_layout()
    return fig


# A panel of distributions(): the record key, how to get a number from each
# entry, the axis label, whether the x axis is logarithmic, whether to count
# only subshowers that deposited energy, and whether the values are counts
_Panel = _namedtuple("_Panel", "key get label log detected_only integer")
DISTRIBUTIONS = [
    _Panel("start_kinetic_energy", None, "start kinetic energy [GeV]", True, False, False),
    _Panel("start_position", lambda v: v[2], "start height (z) [m]", False, False, False),
    _Panel("start_time", None, "start time", False, False, False),
    _Panel("deposity_multiplicity", None, "detector multiplicity", False, False, True),
    _Panel("total_deposit_energy", None, "total deposited energy [GeV]", True, True, False),
    _Panel("max_deposit_energy", None, "max deposit energy [GeV]", True, True, False),
    _Panel("mean_deposit_energy", None, "mean deposit energy [GeV]", True, True, False),
    _Panel(
        "deposit_energy_variance", None, "deposit energy variance [GeV²]", True, True, False
    ),
    _Panel("mean_deposit_radius", None, "mean deposit radius [m]", False, True, False),
    _Panel("max_deposit_radius", None, "max deposit radius [m]", False, True, False),
    _Panel("mean_deposit_offset", None, "mean deposit offset [m]", False, True, False),
    _Panel("deposit_sphericity", None, "deposit sphericity", False, True, False),
]  # fmt: skip


def _panel_values(record, panel):
    values = record[panel.key]
    if panel.get is not None:
        values = [panel.get(value) for value in values]
    values = _np.asarray(values, dtype=float).reshape(-1)
    if panel.detected_only:
        values = values[_np.asarray(record["deposity_multiplicity"]) > 0]
    values = values[_np.isfinite(values)]
    if panel.log:
        values = values[values > 0]
    return values


def distributions(by_group, group_name_prefix=None, title=None):
    """
    A histogram of each observable, with a line for each group, as the
    fraction of the group's subshowers in each bin, so groups of different
    sizes (such as the dataset and a generative model's output) can be
    compared. Deposit observables count only the subshowers that deposited
    energy in some detector; the multiplicity panel shows how many did.
    """
    fig, axarr = _plt.subplots(3, 4, figsize=(18, 12))
    colours = group_colours(by_group)
    handles = {}
    for ax, panel in zip(axarr.ravel(), DISTRIBUTIONS):
        values = {group: _panel_values(by_group[group], panel) for group in colours}
        pooled = [group_values for group_values in values.values() if len(group_values)]
        ax.set_xlabel(panel.label)
        ax.set_ylabel(
            "fraction of subshowers that deposited energy"
            if panel.detected_only
            else "fraction of subshowers"
        )
        if not pooled:
            ax.text(
                0.5, 0.5, "no values", ha="center", va="center", transform=ax.transAxes
            )
            continue
        pooled = _np.concatenate(pooled)
        if panel.integer and pooled.max() - pooled.min() < 60:
            bins = _np.arange(pooled.min() - 0.5, pooled.max() + 1.5)
        else:
            bins = bin_edges(pooled, panel.log)
        for group, colour in colours.items():
            group_values = values[group]
            if not len(group_values):
                continue
            weights = _np.full(len(group_values), 1 / len(group_values))
            *_, patches = ax.hist(
                group_values,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=1.5,
                color=colour,
            )
            handles.setdefault(group, patches[0])
        if panel.log:
            ax.set_xscale("log")
        if panel.integer:
            ax.set_yscale("log")
    labels = []
    for group in handles:
        name = f"{group_name_prefix}{group}" if group_name_prefix else str(group)
        n_subshowers = len(by_group[group]["deposity_multiplicity"])
        labels.append(f"{name} ({n_subshowers} subshowers)")
    fig.suptitle(title or "Distributions of observables", y=0.995)
    fig.tight_layout()
    if handles:
        legend = fig.legend(
            list(handles.values()),
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            bbox_to_anchor=(0.5, 0.955),
            frameon=False,
        )
        # tight_layout leaves room for the title, but not the legend under it
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        fig.subplots_adjust(top=box.y0 - 0.03)
    return fig
