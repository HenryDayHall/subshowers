"""
Methods for reading and handling whole showers
"""

import os as _os
import numpy as _np
import pandas as _pd
import yaml as _yaml


def shower_exists(folder):
    try:
        Reader(folder)
        return True
    except ValueError:
        return False


class Reader:
    """
    Read data created by a modified CORSIKA8 from file
    """

    history_columns = [
        "shower",
        "pdg",
        "px",
        "py",
        "pz",
        "x",
        "y",
        "z",
        "kinetic_energy",
        "time",
        "label",
        "parent_label",
    ]

    def __init__(self, folder: str):
        """
        Parameters
        ----------
        folder : str
            Path to the folder containing the subshowers
        """
        self.folder = folder
        self._history_path = _os.path.join(folder, "history", "history.parquet")
        if not _os.path.exists(self._history_path):
            raise ValueError(f"History file not found at {self._history_path}")
        self.data = _pd.read_parquet(self._history_path)
        self._check_expected_format()
        self._length = len(self.data[self.history_columns[0]])
        if not (self.data["parent_label"] == -1).any():
            root = self.data[self.data["parent_label"] == self.data["label"]].index[0]
            self.data.at[root, "parent_label"] = -1
        self._tree = None

    def _check_expected_format(self):
        """
        Check that the history has the expected columns
        """
        not_found = [
            column for column in self.history_columns if column not in self.data.columns
        ]
        if not_found:
            raise ValueError(
                f"history file at {self._history_path}"
                f" is missing columns: {not_found}"
            )

    @property
    def tree(self):
        """The history's HistoryTree, built on first use."""
        if self._tree is None:
            self._tree = HistoryTree(self.data)
        return self._tree

    def __len__(self) -> int:
        return self._length


def _parent_positions(labels, parent_labels):
    """
    Position of each particle's parent, found by matching its parent_label to
    the other particles' labels; -1 where there's no match.
    """
    n_particles = len(labels)
    if n_particles == 0:
        return _np.zeros(0, dtype=_np.int64)
    if labels[0] == 0 and _np.array_equal(labels, _np.arange(n_particles)):
        # labels are row numbers, as test/fake_data.py writes them
        positions = _np.asarray(parent_labels).astype(_np.int64)
        positions[(positions < 0) | (positions >= n_particles)] = -1
        return positions
    order = _np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    found = _np.searchsorted(sorted_labels, parent_labels)
    found = _np.minimum(found, n_particles - 1)
    return _np.where(sorted_labels[found] == parent_labels, order[found], -1)


class HistoryTree:
    """
    The parent/child structure of a history, worked out once, so that walking
    the tree doesn't mean scanning the whole history at every step.

    Particles are referred to by position (row number) in the history. A
    particle's parent is the particle whose label is its parent_label.
    Particles with no such parent (parent_label -1, a label that isn't in the
    history, or their own label) start trees of their own; the shower's
    primary, `root`, is the first particle with parent_label -1.

    Every particle's subtree is a contiguous slice of `preorder`, so getting
    one is a slice rather than a walk. Labels should be unique, and the
    history must have no loops.
    """

    def __init__(self, data: _pd.DataFrame):
        labels = data["label"].to_numpy()
        parent_labels = data["parent_label"].to_numpy()
        n_particles = len(labels)
        positions = _np.arange(n_particles)
        self.index = data.index
        self._index_is_position = isinstance(data.index, _pd.RangeIndex) and (
            data.index.start == 0 and data.index.step == 1
        )
        self._index_values = None
        roots = _np.flatnonzero(parent_labels == -1)
        self.root = int(roots[0]) if len(roots) else None

        self.parent = _parent_positions(labels, parent_labels)
        self.parent[self.parent == positions] = -1

        # children of each particle, grouped by parent in order
        has_parent = _np.flatnonzero(self.parent >= 0)
        self.n_children = _np.bincount(self.parent[has_parent], minlength=n_particles)
        self.is_leaf = self.n_children == 0
        self._children = has_parent[_np.argsort(self.parent[has_parent], kind="stable")]
        self._child_offsets = _np.concatenate(([0], _np.cumsum(self.n_children)))

        # generations: particles without a parent, then their children,
        # then theirs... Each is grouped by parent, in the order of the last.
        self._generations = [_np.flatnonzero(self.parent < 0)]
        while True:
            children = self.children_of(self._generations[-1])
            if not len(children):
                break
            self._generations.append(children)

        # sizes of subtrees, from the last generation up
        self.subtree_size = _np.ones(n_particles, dtype=_np.int64)
        for parents, children in reversed(self._family_pairs()):
            cumulative = _np.concatenate(([0], _np.cumsum(self.subtree_size[children])))
            ends = _np.cumsum(self.n_children[parents])
            starts = ends - self.n_children[parents]
            self.subtree_size[parents] += cumulative[ends] - cumulative[starts]

        # place in a depth-first ordering: each particle comes just before its
        # subtree, which comes after the subtrees of its earlier siblings
        self._preorder_position = _np.full(n_particles, -1, dtype=_np.int64)
        tops = self._generations[0]
        sizes = self.subtree_size[tops]
        self._preorder_position[tops] = _np.cumsum(sizes) - sizes
        for parents, children in self._family_pairs():
            counts = self.n_children[parents]
            cumulative = _np.concatenate(([0], _np.cumsum(self.subtree_size[children])))
            group_starts = _np.cumsum(counts) - counts
            first = self._preorder_position[parents] + 1 - cumulative[group_starts]
            self._preorder_position[children] = (
                _np.repeat(first, counts) + cumulative[:-1]
            )
        reached = _np.flatnonzero(self._preorder_position >= 0)
        self.preorder = _np.empty(len(reached), dtype=_np.int64)
        self.preorder[self._preorder_position[reached]] = reached

    def _family_pairs(self):
        """(parents, children) for each pair of consecutive generations."""
        return list(zip(self._generations[:-1], self._generations[1:]))

    def children_of(self, positions):
        """Children of the particles at these positions, grouped by parent."""
        counts = self.n_children[positions]
        group_starts = _np.cumsum(counts) - counts
        within_group = _np.arange(counts.sum()) - _np.repeat(group_starts, counts)
        return self._children[
            _np.repeat(self._child_offsets[positions], counts) + within_group
        ]

    def subtree(self, position, leaves_only=False):
        """
        Positions of a particle and everything descended from it, in
        depth-first order; with leaves_only, just the particles without
        children.
        """
        start = self._preorder_position[position]
        if start < 0:
            raise ValueError(f"particle {position} is in a loop of parents")
        members = self.preorder[start : start + self.subtree_size[position]]
        if leaves_only:
            members = members[self.is_leaf[members]]
        return members

    def topmost_marked(self, marked):
        """
        For each particle, the position of the first particle marked true on
        the way down to it from the top of its tree (itself included), or -1.
        """
        owner = _np.where(marked, _np.arange(len(marked)), -1)
        for _, children in self._family_pairs():
            inherited = owner[self.parent[children]]
            owner[children] = _np.where(inherited >= 0, inherited, owner[children])
        return owner

    def labels_of(self, positions):
        """The history's index labels for these positions."""
        if self._index_is_position:
            return _np.asarray(positions)
        if self._index_values is None:
            self._index_values = self.index.to_numpy()
        return self._index_values[positions]

    def positions_of(self, labels):
        """The positions of these index labels in the history."""
        positions = self.index.get_indexer(labels)
        if (positions < 0).any():
            raise KeyError(f"not in the history: {_np.asarray(labels)[positions < 0]}")
        return positions


class ShowerFolders:
    def __init__(self, configs):
        self.configs = configs
        storage_path = configs["storage_path"]
        showers_folder = configs["showers"]["folder"]
        self._base_path = _os.path.join(storage_path, showers_folder)
        self._folders = [
            _os.path.join(self._base_path, folder)
            for folder in configs["showers"]["subfolders"]
        ]
        self.__length = len(self._folders)

    def __getitem__(self, key):
        return self._folders[key]

    def __len__(self):
        return self.__length

    def __iter__(self):
        for folder in self._folders:
            yield folder

    def __str__(self):
        return f"ShowerFiles[{self.__length}]"

    def __repr__(self):
        return f"ShowerFiles[{self._folders}]"


if __name__ == "__main__":
    conf_path = "/home/dayhallh/eas/subshowers/configs/default.yaml"
    with open(conf_path, "r") as f:
        conf = _yaml.safe_load(f)
    print(conf)
    shower_files = ShowerFolders(conf)
    for folder in shower_files:
        print(folder)

    reader = Reader(shower_files[0])
    print(reader.data)
