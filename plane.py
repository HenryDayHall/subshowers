"""
Find the location and energy of particles that reach a detection plane
"""

import numpy as _np
from subshowers import raw_save_load as _raw


def point_on_plane(start_position, direction, height):
    """
    Find the location of intersection of a half line with a plane.

    Parameters
    ----------
    start_position : array (..., 3)
        Last dimension is x, y, z
    direction : array (..., 3)
        Last dimension is dx, dy, dz
    height : float
        Height of the plane, z = height

    Returns
    -------
    intersect_mask : array (...):
        True if the half line intersects the plane
    position : array (..., 2)
        Location of intersection in x, y

    """
    distance_to_plane = (start_position[..., 2] - height) / -direction[..., 2]
    position = (
        start_position[..., [0, 1]]
        + direction[..., [0, 1]] * distance_to_plane[..., None]
    )
    intersect_mask = _np.sign(distance_to_plane) > 0
    return intersect_mask, position


class PlaneDistributions:
    start_columns = [
        "pdg",
        "px",
        "py",
        "pz",
        "x",
        "y",
        "z",
        "kinetic_energy",
        "time",
    ]
    leaf_columns = [
        "pdg",
        "px",
        "py",
        "pz",
        "kinetic_energy",
        "time",
    ]

    def __init__(self, height):
        self.height = height
        self._start_block_length = 100
        self._leaf_block_length = 10_000
        self._start_idx_reached = 0
        self._leaf_idx_reached = 0
        self._print_frequency = 1

    def _init_arrays(self):
        self._subshower_starts = _np.zeros(
            (self._start_block_length, len(self.start_columns))
        )
        self._locations_on_plane = _np.zeros((self._leaf_block_length, 2))
        self._leaf_values = _np.zeros((self._leaf_block_length, len(self.leaf_columns)))
        self._start_indices = _np.zeros((self._start_block_length), dtype=int)

    @property
    def subshower_starts(self):
        return self._subshower_starts[: self._start_idx_reached]

    @property
    def locations_on_plane(self):
        return self._locations_on_plane[: self._leaf_idx_reached]

    @property
    def leaf_values(self):
        return self._leaf_values[: self._leaf_idx_reached]

    @property
    def start_indices(self):
        return self._start_indices[: self._start_idx_reached]

    def append_one_subshower(self, start, locations, leaf_values):
        while self._start_idx_reached <= len(self._subshower_starts):
            self._subshower_starts = _np.append(
                self._subshower_starts,
                _np.zeros((self._start_block_length, len(self.start_columns))),
                axis=0,
            )
        self._subshower_starts[self._start_idx_reached] = start
        self._start_idx_reached += 1
        leaves_to_add = len(locations)
        while leaves_to_add + self._leaf_idx_reached > len(self._locations_on_plane):
            self._locations_on_plane = _np.append(
                self._locations_on_plane,
                _np.zeros((self._leaf_block_length, 2)),
                axis=0,
            )
            self._leaf_values = _np.append(
                self._leaf_values,
                _np.zeros((self._leaf_block_length, len(self.leaf_columns))),
                axis=0,
            )
            self._start_indices = _np.append(
                self._start_indices,
                _np.zeros((self._start_block_length), dtype=int),
                axis=0,
            )
        self._locations_on_plane[
            self._leaf_idx_reached : self._leaf_idx_reached + leaves_to_add
        ] = locations
        self._leaf_values[
            self._leaf_idx_reached : self._leaf_idx_reached + leaves_to_add
        ] = leaf_values
        self._start_indices[
            self._leaf_idx_reached : self._leaf_idx_reached + leaves_to_add
        ] = self._start_idx_reached
        self._leaf_idx_reached += leaves_to_add

    def append_subshowers(self, subshowers):
        n_subshowers = len(subshowers)
        reader = subshowers.reader
        for i, (start_idx, leaf_idxs) in enumerate(subshowers):
            if i % self._print_frequency == 0:
                print(f"Processed {i / n_subshowers:00.0%} of subshowers.", end="\r")

            start = [reader.data[c][start_idx] for c in self.start_columns]
            # work out how reaches which poitn of the plane
            inital_positions = _np.stack(
                [reader.data[c][leaf_idxs] for c in ["x", "y", "z"]], axis=-1
            )
            directions = _np.stack(
                [reader.data[c][leaf_idxs] for c in ["px", "py", "pz"]], axis=-1
            )
            intersect_mask, position = point_on_plane(
                inital_positions, directions, self.height
            )
            locations = position[intersect_mask]
            intersecting_leaves = leaf_idxs[intersect_mask]
            leaf_values = _np.stack(
                [reader.data[c][intersecting_leaves] for c in self.leaf_columns],
                axis=-1,
            )
            self.append_one_subshower(start, locations, leaf_values)

    def _save_dict(self) -> dict:
        irregular_items = {
            "height": self.height,
            "start_columns": self.start_columns,
            "leaf_columns": self.leaf_columns,
            "subshower_starts": self._subshower_starts,
            }

        same_length = {
            "locations_on_plane": self._locations_on_plane,
            "leaf_values": self._leaf_values,
            "start_indices": self._start_indices
        }
        return irregular_items, same_length

    @classmethod
    def from_dict(cls, loaded_dict):
        new_plane = cls(
            float(loaded_dict.pop("height")),
        )
        for key, value in loaded_dict.items():
            if not key.endswith("_columns"):
                key = "_" + key
            setattr(new_plane, key, value)
        return new_plane

    def save(self, path: str):
        if path.endswith(".h5"):
            path = path[:-3]
        irregular_items, same_length = self._save_dict()
        _raw.save(path, same_length, **irregular_items)

    @classmethod
    def load(cls, path: str):
        if path.endswith(".h5"):
            path = path[:-3]
        loaded = _raw.load(path)
        return cls.from_dict(loaded)

