# subshowers

Tools for splitting simulated extensive air showers into *subshowers*, and for
working out what a ground detector array would see of each one.

A shower history produced by a history-retaining build of CORSIKA 8 is a tree of
particles: every particle records its parent, so the whole cascade can be walked
from the primary down to the particles that stop or reach the ground. This
package reads that tree, picks starting points according to some condition
(by default, "kinetic energy has dropped below a cut"), collects the descendants
of each start into a subshower, and can then propagate the resulting particles
onto a detector array to get an energy deposit per detector.

The intended use is building datasets of (subshower start, detector response)
pairs, so that the expensive low-energy part of a cascade can be studied — or
learned — separately from the high-energy part.

## Concepts

| Term | Meaning |
| --- | --- |
| **history** | The full particle tree of one shower, stored as a parquet file. |
| **root** | The primary particle; the only one with `parent_label == -1`. |
| **start** | A particle where a subshower begins, chosen by a `start_condition`. |
| **subshower** | A start particle and all of its descendants. |
| **leaves** | Particles in a subshower with no children — the ones that can hit a detector. |
| **loose ends** | Leaves that never satisfied the start condition, so belong to no subshower. |

## Requirements

- Python 3
- `numpy`, `pandas`, `pyyaml`
- `pyarrow` (or `fastparquet`) for reading history files
- `tables` for the HDF5 output files
- `matplotlib` and `plotly` for the plotting modules

```bash
pip install numpy pandas pyyaml pyarrow tables matplotlib plotly
```

Then run from the repository root, or put it on your `PYTHONPATH`, so that
`import subshowers` resolves.

Generating the input showers is a separate job — see
[Generating data](#generating-data).

## Input data format

Each shower lives in its own folder, containing:

```
<shower folder>/
└── history/
    └── history.parquet
```

The parquet file must have the columns listed in `Reader.history_columns`:

```
shower, pdg, px, py, pz, x, y, z, kinetic_energy, time, label, parent_label
```

`label` identifies a particle, `parent_label` points at its parent, and the root
particle is marked with `parent_label == -1`. If no `-1` is present, `Reader`
treats the self-parenting particle as the root.

## Quick start

### Find the subshowers in one shower

From the command line:

```bash
python -m subshowers.subshowers <shower folder> <energy cut in GeV> [output path]
```

This writes `subshowers_e<cut>.h5` inside the shower folder (unless an output
path is given), holding the shower starts, the particle indices of every
subshower, and the metadata needed to reproduce them.

The same thing from Python:

```python
from subshowers import subshowers

subshowers.run(folder, energy_cut_value=100.0, leaves_only=True, verbose=True)
```

### Build them by hand

Useful when you want a start condition other than an energy cut. A
`start_condition` is any callable taking the history columns as keyword
arguments and returning a bool.

```python
from subshowers.showers import Reader
from subshowers.subshowers import ShowerStarts, Subshowers, energy_cut

reader = Reader(folder)
starts = ShowerStarts(reader, energy_cut(100.0))      # or your own condition
subs = Subshowers(reader, leaves_only=True)
subs.add_subshower(starts)

for start_index, particle_indices in subs:
    ...
```

### Reload a saved set

```python
from subshowers.subshowers import Output

output = Output.load("subshowers_e100.0.h5")
reader = output.subshowers.reader   # the history is reopened alongside it
```

Paths inside the file are stored relative to the file itself, so a dataset can
be moved as long as histories and outputs move together.

### Detector response

`subshowers.detector` builds arrays of sensitive regions (for example
`generate_pierre_auger()` or `generate_at_ground(...)`) and answers whether
particle tracks intersect them. `sampling` puts the pieces together and hands
back one `DetectedSample` per subshower:

```python
from subshowers import detector
from subshowers.sampling import DatasetSource, SubshowerSampler

array = detector.generate_pierre_auger()
sources = [DatasetSource(folder, energy_cut_value=100.0) for folder in folders]
sampler = SubshowerSampler(sources, array, seed=42)

for sample in sampler.iter_epoch(shuffle=True, detected_only=True):
    print(sample.start_kinetic_energy, sample.total_detected_energy)
```

A `DatasetSource` can either load subshowers from a saved `Output` file
(`output_path=...`) or compute them on the fly from a cut
(`energy_cut_value=...`). Histories are large, so sources are opened lazily and
evicted LRU-style; `iter_epoch` visits one source at a time, while `iter_random`
draws uniformly across all of them at the cost of more reloading.

`detector.from_config(config)` builds the array a config's `detectors` section
describes (see [Configuration](#configuration)), and
`datasets.check_subshower_sets(config)` says which of the config's subshower
sets are made, with a `DatasetSource` for each that is.

## Configuration

`configs/default.yaml` describes where a dataset lives, which subshowers to make
from it, and which detector array to build:

```yaml
storage_path: /path/to/data/
showers:
  folder: examples
  subfolders: [example_10e4_photon_hist1, ...]
subshowers:
  energy_cuts: [100, 500, 900]   # GeV, one set of subshowers per shower and cut
  leaves_only: true
detectors:
  generator:
    name: generate_at_ground  # a generate_ function in subshowers/detector.py
    total: 20                 # and its arguments
    ...
    seed: 0
```

The detector generators are `generate_at_ground` (random triangles; give it a
`seed` so it's the same array every time), `generate_from_csv`, and
`generate_pierre_auger`, the Pierre Auger surface detector:

```yaml
detectors:
  generator:
    name: generate_pierre_auger
    for_array: stop  # the stations still running; or sd1500, or sd750
```

Leaving out `for_array` gives every station listed, including some since
replaced at the same place, where a particle is counted twice.

A `ground` section, which can be left out, describes the ground for the
ground-level observables (see
[Ground-level observables](#ground-level-observables)).

`ShowerFolders` turns that into a list of shower folders:

```python
import yaml
from subshowers.showers import ShowerFolders

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
folders = ShowerFolders(config)
```

Copy the file and edit the paths rather than editing the default in place.

## Repository layout

```
subshowers/
├── showers.py         Reader for history files; HistoryTree; ShowerFolders from a config
├── subshowers.py      Shower starts, subshower extraction, saving and loading
├── detector.py        Detector arrays and track/region intersection
├── responses.py       Energy deposited per detector; Responses, which stores it for many subshowers
├── sampling.py        DatasetSource / SubshowerSampler over many shower folders
├── datasets.py        Which of the subshower sets a config asks for are made
├── observables.py     Observables of detector responses, and figures of them
├── population_plotting.py  Histograms and scatter plots for those figures
├── ground.py          What subshowers put on the ground; GroundSummaries, which stores it
├── ground_observables.py  Ground-level observables (lateral distributions, spectra, ...), and figures of them
├── plane.py           Particles crossing a horizontal detection plane
├── sample_plotting.py Matplotlib and plotly views of showers and responses
├── raw_save_load.py   Small HDF5 + yaml save/load helper
└── generation.py      Plans and submits the SLURM jobs that make showers and subshowers
corsika_interactions/  Scripts to install a history-retaining CORSIKA 8 and generate showers
configs/               Example configuration
scripts/               Notebooks, analysis scripts and data generation scripts
test/                  pytest suite, plus generators for fake histories
```

## Installing CORSIKA

`corsika_interactions/` installs a modified CORSIKA 8 (the `histories_1` branch
of `HenryDayHall/CORSIKA8`) into a conda environment:

```bash
corsika_interactions/setup_if_needed.sh <install directory> [<env name>]
```

The install builds without FLUKA. Shower jobs look for it in the folder above
the config's `storage_path`.

## Generating data

One command submits every SLURM job needed to make the data a config
describes:

```bash
python scripts/generate_data_from_config.py configs/default.yaml [--dry-run]
```

It checks what's already made or queued, then submits

- shower jobs for missing showers, one array job per energy;
- subshower jobs for missing energy cuts, one array job per shower, running
  `subshowers.run` in the `corsika` conda environment. Jobs for showers still
  being made wait for them (`--dependency=afterok`), and are cancelled if the
  shower job fails.

`corsika_interactions/generation_from_config.py` and
`scripts/subshower_generation_from_config.py` take the same arguments, and
submit only showers or only subshowers. `--dry-run` writes the job scripts
without submitting them.

Job scripts go in `jobs/` in the showers folder, with `submitted_jobs.yaml`,
a record of what each job makes. Along with `squeue`, that keeps queued work
from being submitted twice. Showers that were started but never finished are
left alone; delete the folder to make one again.

Each job's time limit is twice its estimated run time, rounded up to whole
hours, from 1 hour up to the 7-day maximum of the `maxcpu` partition. The
estimates are in `subshowers/generation.py`:

- shower run time follows energy (`SHOWER_MINUTES`), starting from 18.3
  minutes for a 1e5 GeV photon;
- subshower run time grows in step with the number of particles in the
  history (`subshower_minutes`): about 7 seconds and 3 GB of memory for 10
  million particles. The number is read from finished histories (with
  pyarrow) and scaled by energy for showers not made yet.

Jobs estimated to need more than 7 days aren't submitted.

## Plotting observables

```bash
python scripts/plot_observables_from_config.py configs/default.yaml
```

samples every subshower set the config asks for with the config's detector
array, and plots observables of what the array sees: start conditions against
deposited energy, multiplicity, and the size and shape of the deposits; and
of what the subshowers put on the ground (see
[Ground-level observables](#ground-level-observables)). It
lists the sets that are missing, and why, and plots the rest. The plots and a
`report.txt` go in `plots/<run_name>` in the showers folder (or
`--output-dir`):

- `e<cut>/` for each energy cut, with the subshowers grouped by the primary
  energy of their shower, by the particle that starts them, and for electrons
  and positrons alone, plus a few examples of the detectors hit;
- `all_cuts/`, comparing the cuts.

The `distributions_*` figures normalise each group to its own number of
subshowers, so groups of different sizes can be compared.

The detector responses are slow to work out, so each set's are saved in
`responses/` in the output folder, and used again while the set, the detector
and the options stay the same. `--max-subshowers N` uses N subshowers at random
from each set, for a quicker look; `--detected-only` plots only the subshowers
that deposited energy; `--recompute` works the responses out again.

Saved responses are what a generative model's subshowers can be compared with.
Put the model's subshowers through the same array, with `start_pdg`,
`start_kinetic_energy`, `start_time`, `start_position`, `start_direction` and
`energy_per_detector` for each, and plot the two together:

```python
from subshowers import detector, observables
from subshowers.responses import Responses

array = detector.from_config(config)
dataset = Responses.concatenate(Responses.load(path) for path in saved_paths)
model = Responses.from_samples(model_samples, len(array), source="model")
fig = observables.distributions(
    {
        "dataset": observables.record_of(dataset, array),
        "model": observables.record_of(model, array),
    }
)
```

### Ground-level observables

The same subshowers are also measured where they reach the ground, for the
observables air shower simulations are compared with at ground level (as
when comparing versions of CORSIKA, or CORSIKA with and without FLUKA). For
each grouping, the plotting script draws

| Figure | What it shows |
| --- | --- |
| `ground_lateral_<frame>_axis_*` | Lateral distributions: particles per m² of e±, γ, μ± and hadrons against distance from the axis, and the energy the detectors read per m² of detector |
| `ground_spectra_*` | Kinetic energy spectra of each kind of particle reaching the ground |
| `ground_numbers_*` | How many of each reach the ground: the fraction of subshowers with at least N, the mean against start energy, and the relative fluctuations (as for the muon number) |
| `ground_timing_<frame>_axis_*` | Arrival delays after a plane front moving along the axis at the speed of light, and their spread (the thickness of the shower front), against distance from the axis |
| `ground_reference_<frame>_axis_*` | Readings at a reference distance from the axis, like the Telescope Array's S800: the detector signal there, and the densities of e± and μ± there against start energy |

Lines are means over a group's subshowers, with a band of the standard error
of the mean; arrival times are averaged over the particles instead.

Each figure is drawn twice: of every particle reaching the ground, in
`ground_*`, and of only the particles that hit a detector, in
`ground_detected_*` (e.g. `ground_detected_lateral_subshower_axis_by_pdg.png`).
The detected particles are counted the same way, in the same bins and per
m² of the same area (of the shower plane, not of the detectors), so each
detected curve is the part of the full one that the array observes. These
are the metrics to compare a generative model against when the model only
makes particles that reach detectors.

The particles a subshower puts on the ground are its leaves, carried down in
straight lines as the detector sampler carries them, and slowed by their
mass. CORSIKA 8 writes particles to the history before deciding whether to
track them, so leaves below its tracking cuts (0.5 MeV for e± and γ, 0.3
GeV for muons and hadrons) aren't counted; nor are neutrinos. Energies are
as the particles were made: losses on the way down aren't known. Times in
the history are in seconds.

Distances and delays are measured about each subshower's own axis (the line
through its start particle along its direction), or with `--frame shower`
about the shower's axis, so that each subshower's lateral distribution is
its share of what an array around the shower's core reads. Distances are in
the shower plane, and densities are per m² of it.

The config's `ground` section sets the height of the ground (0 is CORSIKA's
observation level), the kinetic energies below which particles aren't
counted, and the reference distance in each frame. Left out, it's

```yaml
ground:
  height: 0.0
  min_kinetic_energy: {electrons: 5.0e-4, photons: 5.0e-4, muons: 0.3, hadrons: 0.3}
  reference_distance: {subshower: 100.0, shower: 800.0}
```

A particle hits a detector if its straight line from where it was made
crosses one, as for the detector responses, which record which leaves did.
For the detected figures it's placed, like every particle, where its line
meets the ground: where it meets the detector if the detectors lie on the
ground, as `generate_at_ground` puts them. For an array above it, set the
ground's `height` to the detectors'. Deposits count every leaf that hits a
detector, but the detected figures leave out those below the tracking cuts,
and neutrinos, so a few subshowers that deposit energy may put no counted
particle on a detector.

Each subshower is summarised by histograms with fixed bins, for both frames,
so `--frame` only changes the plots. Each set's summaries are saved in
`ground/` in the output folder, and used again while the set, the sampling
options and the `ground` section stay the same. Those of the detected
particles are saved in `ground_detected/`, and are worked out again whenever
the responses are measured again, too. Responses saved before they recorded
which leaves hit a detector are measured again.

A generative model's subshowers are compared in the same way. Give each the
start values, as for responses, and the particles it puts on the ground:
`ground_pdg`, `ground_x`, `ground_y`, `ground_kinetic_energy` and, if the
model gives them, `ground_time` (in seconds). With `primary_position`,
`primary_direction` and `primary_time`, the shower's axis can be used too.
For a model that only makes the particles that reach detectors, say so with
`particles="detected"`, and compare it with the dataset's detected
particles:

```python
from subshowers import ground_observables
from subshowers.ground import GroundSummaries

# saved_paths: the dataset's files in ground_detected/
dataset = GroundSummaries.concatenate(GroundSummaries.load(path) for path in saved_paths)
model = GroundSummaries.from_samples(
    model_samples, dataset.settings, source="model", particles="detected"
)
by_group = {
    "dataset": ground_observables.record_of(dataset, "subshower"),
    "model": ground_observables.record_of(model, "subshower"),
}
fig = ground_observables.lateral_distributions(
    by_group,
    "subshower",
    reference="dataset",
    reference_distance=100.0,
    particles="detected",
)
```

Every figure function takes `particles`, `"all"` (the default) or
`"detected"`. A record can hold both kinds: those of all particles are
under keys starting `ground_`, and those of detected particles under keys
starting `detected_`. A group without the kind a figure draws is listed in
its legend, but has no lines. `reference` adds panels of each group's ratio
to it under the lateral distributions and energy spectra. Passing
`record_of` each population's `Responses`, and the array, adds the detector
signal.

## Plotting showers

```bash
python -m subshowers.sample_plotting <saved subshowers .h5> [x_axis] [y_axis]
```

writes an interactive plotly HTML file showing the shower and a random selection
of subshowers. `sample_plotting` also has matplotlib helpers
(`show_pierre_auger`, `show_subshower`) for looking at detector responses.

## Tests

```bash
cd test
pytest
```

`test/make_fake_datasets.py` grows small synthetic cascades in the expected
folder layout, which is handy for trying things out without a real CORSIKA run.

## Licence

Apache License 2.0 — see [`LICENCE`](LICENCE).
