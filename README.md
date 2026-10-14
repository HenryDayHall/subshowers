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
deposited energy, multiplicity, and the size and shape of the deposits. It
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
