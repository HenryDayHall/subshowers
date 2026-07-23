import numpy as _np
from matplotlib import pyplot as _plt
from subshowers.population_plotting import scatter_with_marginals, spectrum_histogram


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


def get_samples_per_source(sampler, array):
    per_source = {}
    n_total, n_detected = 0, 0
    for sample in sampler.iter_epoch(shuffle=True):
        n_total += 1
        source_name = sample.source.name
        if source_name not in per_source:
            per_source[source_name] = {
                value: [] for value in conditioning_values + output_values
            }
        record = per_source[source_name]
        for name in conditioning_values:
            record[name].append(getattr(sample, name))
        get_outputs(sample, array, record)

    print(f"Epoch complete: {n_detected}/{n_total} subshowers deposited energy")
    return per_source


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


def observables_A(by_group, **kwargs):
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

    fig.suptitle("Subshower start conditions vs detected response", y=0.995)
    return fig


def observables_B(by_group, **kwargs):
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

    fig.suptitle("Subshower start energy vs deposit shape", y=0.995)
    return fig


def observables_C(by_group, **kwargs):
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

    fig.suptitle("Average per-sample deposit spectra", y=0.995)
    _plt.tight_layout()
    return fig
