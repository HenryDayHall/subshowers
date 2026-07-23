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
import os

import matplotlib.pyplot as plt
import numpy as np

from subshowers import detector
from subshowers.showers import shower_exists
from subshowers.sampling import DatasetSource, SubshowerSampler
from subshowers.sample_plotting import show_subshower
from subshowers.observables import (
    get_samples_per_source,
    filter_pdgs,
    gather_by_pdg,
    observables_A,
    observables_B,
    observables_C,
)

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

per_source = get_samples_per_source(sampler, array)
# gather every subshower across all sources, grouped by its start_pdg so
# that the plots below use one colour per particle type rather than per file

fig = observables_A(per_source)
plt.savefig(f"{base_dir}/population_statistics_A_by_source.png", dpi=130)

fig = observables_B(per_source)
plt.savefig(f"{base_dir}/population_statistics_B_by_source.png", dpi=130)

fig = observables_C(per_source)
plt.savefig(f"{base_dir}/population_statistics_C_by_source.png", dpi=130)

per_source_electrons_and_positrons = filter_pdgs(
    per_source, keep_pdgs=[11, -11]
)

fig = observables_A(per_source_electrons_and_positrons)
plt.savefig(
    f"{base_dir}/population_statistics_A_by_source_electrons_and_positrons.png",
    dpi=130,
)

fig = observables_B(per_source_electrons_and_positrons)
plt.savefig(
    f"{base_dir}/population_statistics_B_by_source_electrons_and_positrons.png",
    dpi=130,
)

fig = observables_C(per_source_electrons_and_positrons)
plt.savefig(
    f"{base_dir}/population_statistics_C_by_source_electrons_and_positrons.png",
    dpi=130,
)

by_pdg = gather_by_pdg(per_source)

fig = observables_A(by_pdg, group_name_prefix="pdg")
plt.savefig(f"{base_dir}/population_statistics_A.png", dpi=130)

fig = observables_B(by_pdg, group_name_prefix="pdg")
plt.savefig(f"{base_dir}/population_statistics_B.png", dpi=130)

fig = observables_C(by_pdg, group_name_prefix="pdg")
plt.savefig(f"{base_dir}/population_statistics_C.png", dpi=130)
print("Wrote population statistics")
