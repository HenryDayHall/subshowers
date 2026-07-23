"""
Demo of subshowers.sampling.SubshowerSampler.

1. Builds two DatasetSources (two shower folders, subshowers from an
   energy cut) and one detector array.
2. Plots the detected portion of a few randomly sampled subshowers, in
   the style of the final cell of scripts/ViewDetectorArrays.ipynb
   (detector centres coloured by deposited energy, LogNorm, zoomed to
   the active region with detector.center_and_size).
3. Uses one full shuffled epoch of the sampler to build a population
   statistic: the distribution of total detected energy per subshower,
   split by dataset.
"""
import matplotlib
import os

import matplotlib.pyplot as plt
import numpy as np

from subshowers import detector
from subshowers.showers import shower_exists
from subshowers.sampling import DatasetSource, SubshowerSampler
from subshowers.sample_plotting import show_subshower

from subshowers_test import fake_data

# ------------------------------------------------------------------ setup
base_dir = "/data/dust/user/dayhallh/eas/data/sampling_demo"

ENERGY_CUT = 10_000.0  # GeV, subshowers start where KE drops below this
specs = {  # folder -> (seed, root energy [GeV], root height [m])
    "example_photon_hist1": (1, 4.0e4, 170_000.0),
    "example_photon_hist2": (2, 7.0e4, 190_000.0),
}

FOLDERS = [os.path.join(base_dir, name) for name in specs]
FOLDERS = []

print("Generating fake showers:")
for name, (seed, energy, height) in specs.items():
    folder = os.path.join(base_dir, name)
    FOLDERS.append(folder)
    if shower_exists(folder):
        print(f"{name} already exists. No need to fake more data.")
        continue
    rng = np.random.default_rng(seed)
    data = fake_data.grow_cascade(rng, energy, height)
    fake_data.fake_history(folder, data)
    print(
        f"{name}: {len(data['kinetic_energy'])} particles, "
        f"{(data['kinetic_energy'] < 100).sum()} below 100 GeV"
    )
    del data


array = detector.generate_pierre_auger()
sources = [DatasetSource(folder, energy_cut_value=ENERGY_CUT) for folder in FOLDERS]
sampler = SubshowerSampler(sources, array, seed=42, max_open=2)
print(f"Sampler covers {len(sampler)} subshowers over {len(sources)} datasets")


# --------------------------------------------- plotting as in the notebook
def plot_sample(sample, ax, pad_factor=2.5):
    """One panel: the detected portion of one subshower."""
    show_subshower(
        sample.start_kinetic_energy,
        sample.energy_per_detector,
        array=array,
        axis=ax,
        initial_position=sample.start_position,
    )
    plt.title(
        f"{sample.source.name}  start KE {sample.start_kinetic_energy:.3g} GeV\n"
        f"{sample.n_detected_leaves}/{sample.n_leaves} leaves detected, "
        f"{sample.n_hit_detectors} detectors, "
        f"E_det {sample.total_detected_energy:.3g} GeV",
        fontsize=9,
    )


# a few random samples that actually hit the array, as in the notebook plot
rows = 2
cols = 2
fig, axarr = plt.subplots(rows, cols, figsize=(13, 11))
demo_samples = []
max_samples = rows * cols
for sample in sampler.iter_random(detected_only=True):
    print(f"Iterating samples; {len(demo_samples)/max_samples:.0%}", flush=True)
    demo_samples.append(sample)
    if len(demo_samples) >= max_samples:
        break
print()

for panel, sample in enumerate(demo_samples, start=1):
    ax = axarr.ravel()[panel - 1]
    plot_sample(sample, ax)

plt.suptitle("Detected portion of randomly sampled subshowers", y=0.995)
plt.tight_layout()
plt.savefig(f"{base_dir}/sampled_subshowers.png", dpi=130)
plt.close(fig)
print("Wrote sampled_subshowers.png")

# ------------------------------------------ population statistic (1 epoch)
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

# radial bin edges (metres) for the deposited-energy radial profile



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
        output_dict["deposit_radial_spectrum"].append(
            []
        )
        output_dict["mean_deposit_offset"].append(np.nan)
        output_dict["max_deposit_radius"].append(0.0)
        output_dict["mean_deposit_radius"].append(0.0)
        output_dict["deposit_sphericity"].append(np.nan)
        output_dict["deposit_energy_variance"].append(0.0)
        return output_dict

    # energy-weighted centroid of the deposits
    weights = nonzero_deposits / total_energy
    centroid = np.sum(hit_positions * weights[:, None], axis=0)

    # offset of the deposit centroid from the subshower start (x, y)
    mean_deposit_offset = np.linalg.norm(centroid - start_position[:2])
    output_dict["mean_deposit_offset"].append(mean_deposit_offset)

    # radial spread of the deposits about their centroid
    relative = hit_positions - centroid
    radii = np.linalg.norm(relative, axis=-1)

    # deposited-energy radial profile about the centroid
    output_dict["deposit_radial_spectrum"].append(radii)

    output_dict["max_deposit_radius"].append(float(radii.max()))
    output_dict["mean_deposit_radius"].append(float(np.sum(radii * weights)))

    # sphericity from the energy-weighted 2D spatial covariance:
    # ratio of smaller to larger eigenvalue (1 = round, 0 = linear)
    covariance = (relative * weights[:, None]).T @ relative
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.max() > 0:
        sphericity = float(eigenvalues.min() / eigenvalues.max())
    else:
        sphericity = np.nan
    output_dict["deposit_sphericity"].append(sphericity)

    # variance of the individual detector deposits
    output_dict["deposit_energy_variance"].append(float(np.var(nonzero_deposits)))

    return output_dict


per_source = {
    source.name: {value: [] for value in conditioning_values + output_values}
    for source in sources
}
n_total, n_detected = 0, 0
for sample in sampler.iter_epoch(shuffle=True):
    n_total += 1
    record = per_source[sample.source.name]
    for name in conditioning_values:
        record[name].append(getattr(sample, name))
    get_outputs(sample, array, record)

print(f"Epoch complete: {n_detected}/{n_total} subshowers deposited energy")


# gather every subshower across all sources, grouped by its start_pdg so
# that the plots below use one colour per particle type rather than per file
def gather_by_pdg(per_source, keys):
    """
    Return {pdg: {key: np.ndarray of values}} pooling all sources.
    """
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
                store[key] = np.asarray(store[key])
            except ValueError:
                pass
    return by_pdg


def scatter_with_marginals(
    gridspec,
    by_pdg,
    x_key,
    y_key,
    x_get,
    y_get,
    x_label,
    y_label,
    x_log=True,
    y_log=False,
):
    """
    Draw one scatter panel (start KE on x) with marginal histograms, one
    colour per start_pdg, into the sub-gridspec ``gridspec``.
    """
    inner = gridspec.subgridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.05,
        hspace=0.05,
    )
    ax = fig.add_subplot(inner[1, 0])
    ax_topx = fig.add_subplot(inner[0, 0], sharex=ax)
    ax_righty = fig.add_subplot(inner[1, 1], sharey=ax)
    ax_topx.tick_params(labelbottom=False)
    ax_righty.tick_params(labelleft=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    pdgs = sorted(by_pdg)
    colours = plt.cm.tab10(np.linspace(0, 1, len(pdgs)))
    pdg_colours = dict(zip(pdgs, colours))

    all_x, all_y = [], []
    for pdg in pdgs:
        record = by_pdg[pdg]
        xs = np.asarray([x_get(v) for v in record[x_key]], dtype=float)
        ys = np.asarray([y_get(v) for v in record[y_key]], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        ax.scatter(xs, ys, s=8, alpha=0.5, color=pdg_colours[pdg], label=f"pdg {pdg}")
        all_x.append(xs)
        all_y.append(ys)

    if not all_x:
        return ax
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_bins = (
        np.geomspace(all_x.min(), all_x.max(), 40)
        if x_log
        else np.linspace(all_x.min(), all_x.max(), 40)
    )
    y_bins = (
        np.geomspace(all_y.min(), all_y.max(), 40)
        if y_log and all_y.min() > 0
        else np.linspace(all_y.min(), all_y.max(), 40)
    )
    for pdg in pdgs:
        record = by_pdg[pdg]
        xs = np.asarray([x_get(v) for v in record[x_key]], dtype=float)
        ys = np.asarray([y_get(v) for v in record[y_key]], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        ax_topx.hist(xs, bins=x_bins, histtype="step", color=pdg_colours[pdg])
        ax_righty.hist(
            ys,
            bins=y_bins,
            histtype="step",
            color=pdg_colours[pdg],
            orientation="horizontal",
        )

    if x_log:
        ax.set_xscale("log")
    if y_log and all_y.min() > 0:
        ax.set_yscale("log")
    ax.legend(fontsize=7)
    return ax


by_pdg = gather_by_pdg(
    per_source,
    [
        "start_pdg",
        "start_kinetic_energy",
        "start_time",
        "start_position",
        "total_deposit_energy",
        "deposity_multiplicity",
        "mean_deposit_radius",
        "max_deposit_radius",
        "deposit_sphericity",
        "mean_deposit_offset",
        "deposit_energy_spectrum",
        "deposit_radial_spectrum",
    ],
)

fig = plt.figure(figsize=(14, 12))
outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)

scatter_with_marginals(
    outer[0, 0],
    by_pdg,
    "start_kinetic_energy",
    "start_time",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "start time",
)
scatter_with_marginals(
    outer[0, 1],
    by_pdg,
    "start_kinetic_energy",
    "start_position",
    lambda v: v,
    lambda v: v[2],
    "start kinetic energy [GeV]",
    "start height (z) [m]",
)
scatter_with_marginals(
    outer[1, 0],
    by_pdg,
    "start_kinetic_energy",
    "total_deposit_energy",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "total deposited energy [GeV]",
    y_log=True,
)
scatter_with_marginals(
    outer[1, 1],
    by_pdg,
    "start_kinetic_energy",
    "deposity_multiplicity",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "detector multiplicity",
)

fig.suptitle("Subshower start conditions vs detected response", y=0.995)
plt.savefig(f"{base_dir}/population_statistics_A.png", dpi=130)

fig = plt.figure(figsize=(14, 12))
outer = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)

scatter_with_marginals(
    outer[0, 0],
    by_pdg,
    "start_kinetic_energy",
    "mean_deposit_radius",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "mean deposit radius [m]",
)
scatter_with_marginals(
    outer[0, 1],
    by_pdg,
    "start_kinetic_energy",
    "max_deposit_radius",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "max deposit radius [m]",
)
scatter_with_marginals(
    outer[1, 0],
    by_pdg,
    "start_kinetic_energy",
    "deposit_sphericity",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "deposit sphericity",
)
scatter_with_marginals(
    outer[1, 1],
    by_pdg,
    "start_kinetic_energy",
    "mean_deposit_offset",
    lambda v: v,
    lambda v: v,
    "start kinetic energy [GeV]",
    "mean deposit offset [m]",
)


fig.suptitle("Subshower start energy vs deposit shape", y=0.995)
plt.savefig(f"{base_dir}/population_statistics_B.png", dpi=130)


def spectrum_histogram(ax, by_pdg, key, x_label, log_x=False, combined=False):
    """
    Overlay, one line per start_pdg, the average per-sample frequency of a
    per-sample spectrum (a list/array of values for each subshower).  The
    bin range spans every value seen across all samples and pdgs.  A shaded
    band shows the 0.25 and 0.75 quantiles of the per-sample bin counts.

    With ``combined=True`` every pdg is pooled into a single line.
    """
    if combined:
        merged = {key: [s for store in by_pdg.values() for s in store[key]]}
        by_pdg = {"all": merged}
    pdgs = sorted(by_pdg)
    colours = plt.cm.tab10(np.linspace(0, 1, len(pdgs)))
    pdg_colours = dict(zip(pdgs, colours))

    # per-pdg list of per-sample value arrays, and the pooled values
    per_sample = {}
    pooled = {}
    all_values = []
    for pdg in pdgs:
        spectra = [np.asarray(s, dtype=float).ravel() for s in by_pdg[pdg][key]]
        per_sample[pdg] = spectra
        values = np.concatenate(spectra) if spectra else np.array([])
        pooled[pdg] = values
        if values.size:
            all_values.append(values)

    if not all_values:
        ax.set_xlabel(x_label)
        ax.set_ylabel("count per sample")
        return

    all_values = np.concatenate(all_values)
    lo, hi = all_values.min(), all_values.max()
    if log_x and lo > 0:
        bins = np.geomspace(lo, hi, 40)
    else:
        bins = np.linspace(lo, hi, 40)
    centres = 0.5 * (bins[:-1] + bins[1:])

    for pdg in pdgs:
        spectra = per_sample[pdg]
        if not spectra:
            continue
        # bin counts for each sample: (n_samples, n_bins)
        per_sample_counts = np.array(
            [np.histogram(values, bins=bins)[0] for values in spectra], dtype=float
        )
        mean_counts = per_sample_counts.mean(axis=0)
        lower, upper = np.quantile(per_sample_counts, [0.25, 0.75], axis=0)

        colour = pdg_colours[pdg]
        label = "all pdg" if pdg == "all" else f"pdg {pdg}"
        ax.step(centres, mean_counts, where="mid", color=colour, label=label)
        ax.fill_between(
            centres, lower, upper, step="mid", color=colour, alpha=0.2
        )

    if log_x and lo > 0:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("count per sample")
    ax.legend(fontsize=7)
    ax.semilogy()


fig, axarr = plt.subplots(2, 2, figsize=(14, 11))
(ax_energy, ax_radial), (ax_energy_all, ax_radial_all) = axarr

spectrum_histogram(
    ax_energy,
    by_pdg,
    "deposit_energy_spectrum",
    "deposit energy [GeV]",
    log_x=True,
)
ax_energy.set_title("Deposit energy spectrum (per pdg)")
spectrum_histogram(
    ax_radial,
    by_pdg,
    "deposit_radial_spectrum",
    "deposit radius bin [m]",
)
ax_radial.set_title("Deposit radial spectrum (per pdg)")

spectrum_histogram(
    ax_energy_all,
    by_pdg,
    "deposit_energy_spectrum",
    "deposit energy [GeV]",
    log_x=True,
    combined=True,
)
ax_energy_all.set_title("Deposit energy spectrum (all pdg)")
spectrum_histogram(
    ax_radial_all,
    by_pdg,
    "deposit_radial_spectrum",
    "deposit radius bin [m]",
    combined=True,
)
ax_radial_all.set_title("Deposit radial spectrum (all pdg)")

fig.suptitle("Average per-sample deposit spectra", y=0.995)
plt.tight_layout()
plt.savefig(f"{base_dir}/population_statistics_C.png", dpi=130)
print("Wrote population statistics")
