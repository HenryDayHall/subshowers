import os
import tempfile
import warnings

import numpy as np

from subshowers import plane, subshowers
from subshowers.showers import Reader
from fake_data import fake_history, grow_cascade


def test_plane_distributions():
    with tempfile.TemporaryDirectory() as folder:
        data = grow_cascade(np.random.default_rng(1), 4.0e4, 170_000.0)
        fake_history(folder, data)
        reader = Reader(folder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            starts = subshowers.ShowerStarts(reader, subshowers.energy_cut(100.0))
        subs = subshowers.Subshowers(reader, leaves_only=True)
        subs.add_subshower(starts)

        height = 1000.0  # below every particle, which test/fake_data.py stops at 1600
        distributions = plane.PlaneDistributions(height)
        distributions.append_subshowers(subs)

        # enough to grow past the first block of starts and of leaves
        assert len(distributions.subshower_starts) == len(subs) > 100
        n_leaves = len(distributions.locations_on_plane)
        assert n_leaves > 10_000
        assert len(distributions.leaf_values) == n_leaves
        assert len(distributions.start_indices) == n_leaves

        # each subshower's leaves that reach the plane are labelled with its row
        for row, (start, leaves) in enumerate(subs):
            positions = np.stack([reader.data[c][leaves] for c in "xyz"], axis=-1)
            directions = np.stack(
                [reader.data[c][leaves] for c in ["px", "py", "pz"]], axis=-1
            )
            reaches, _ = plane.point_on_plane(positions, directions, height)
            assert np.sum(distributions.start_indices == row) == reaches.sum()
            assert (
                distributions.subshower_starts[row][
                    plane.PlaneDistributions.start_columns.index("kinetic_energy")
                ]
                == reader.data["kinetic_energy"][start]
            )

        path = os.path.join(folder, "plane.h5")
        distributions.save(path)
        loaded = plane.PlaneDistributions.load(path)
        assert loaded.height == height
        for name in [
            "subshower_starts",
            "locations_on_plane",
            "leaf_values",
            "start_indices",
        ]:
            assert np.array_equal(getattr(loaded, name), getattr(distributions, name))
