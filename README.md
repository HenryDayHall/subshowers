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
[`corsika_interactions/`](#repository-layout).

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

## Configuration

`configs/default.yaml` describes where a dataset lives and which detector array
to build:

```yaml
storage_path: /path/to/data/
showers:
  folder: examples
  subfolders: [example_10e4_photon_hist1, ...]
detectors:
  generator:
    name: generate_at_ground
    ...
```

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
├── showers.py         Reader for history files; ShowerFolders from a config
├── subshowers.py      Shower starts, subshower extraction, saving and loading
├── detector.py        Detector arrays and track/region intersection
├── responses.py       Energy deposited per detector by a set of particles
├── sampling.py        DatasetSource / SubshowerSampler over many shower folders
├── plane.py           Particles crossing a horizontal detection plane
├── sample_plotting.py Matplotlib and plotly views of showers and responses
└── raw_save_load.py   Small HDF5 + yaml save/load helper
corsika_interactions/  Scripts to install a history-retaining CORSIKA 8 and generate showers
configs/               Example configuration
scripts/               Notebooks and analysis scripts
test/                  pytest suite, plus generators for fake histories
```

## Generating showers

`corsika_interactions/` installs a modified CORSIKA 8 (the `histories_1` branch
of `HenryDayHall/CORSIKA8`) into a conda environment and submits shower
generation jobs:

```bash
corsika_interactions/setup_if_needed.sh <install directory> [<env name>]
python corsika_interactions/generation_from_config.py configs/default.yaml
```

The install builds without FLUKA.

## Plotting

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
