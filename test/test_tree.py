"""
Checks that HistoryTree and the subshower code built on it give the same
results as the original tree walk, which scanned the whole history at every
step, on random cascades laid out in several ways.
"""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from subshowers import subshowers
from subshowers.showers import HistoryTree, Reader
from fake_data import grow_cascade, fake_history, mock_history_dict_2

# ------------------------------------------------- the original implementation


def reference_starts(data, start_condition):
    root = data.index[data["parent_label"] == -1][0]
    if start_condition(**data.loc[root]):
        return [root], []
    starts, loose_ends, stack = [], [], [root]
    while stack:
        parent = stack.pop()
        child_indices = data.index[data["parent_label"] == data.at[parent, "label"]]
        child_values = data.loc[child_indices]
        if child_values.empty:
            loose_ends.append(parent)
            continue
        is_start = start_condition(**child_values)
        starts += list(child_indices[is_start])
        stack += list(child_indices[~is_start])
    return starts, loose_ends


def reference_subshower(data, start_index, leaves_only):
    subshower, stack = [], [start_index]
    while stack:
        parent = stack.pop()
        child_indices = data.index[data["parent_label"] == data.at[parent, "label"]]
        if child_indices.empty or not leaves_only:
            subshower.append(parent)
        stack += list(child_indices)
    return subshower


# ------------------------------------------------------------------ histories


def cascade(seed, energy=2.5e3):
    return pd.DataFrame(grow_cascade(np.random.default_rng(seed), energy, 170_000.0))


def as_generated(data):
    return data


def shuffled(data):
    """Rows out of order, so labels aren't row numbers."""
    return data.sample(frac=1, random_state=2).reset_index(drop=True)


def shuffled_with_index(data):
    """Rows out of order and an index that isn't 0, 1, 2..."""
    data = shuffled(data)
    data.index = 10 * data.index + 3
    return data


def offset_labels(data):
    """Labels that aren't row numbers, with the root still marked -1."""
    data = data.copy()
    data["label"] += 1000
    data.loc[data["parent_label"] >= 0, "parent_label"] += 1000
    return data


LAYOUTS = [as_generated, shuffled, shuffled_with_index, offset_labels]


class DataReader:
    """Stands in for Reader, with the history in memory."""

    def __init__(self, data):
        self.data = data
        self.folder = "in memory"


def pdg_and_energy(pdg, kinetic_energy, **_):
    return (abs(pdg) == 11) & (kinetic_energy < 500)


# ---------------------------------------------------------------------- tests


@pytest.mark.parametrize("layout, seed", list(zip(LAYOUTS, [1, 2, 3, 4])))
def test_matches_original(layout, seed):
    data = layout(cascade(seed))
    root_energy = data["kinetic_energy"].max()
    conditions = [subshowers.energy_cut(cut) for cut in (-1.0, 10.0, 100.0, 1000.0)] + [
        subshowers.energy_cut(2 * root_energy),
        pdg_and_energy,
    ]
    for condition in conditions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            starts = subshowers.ShowerStarts(DataReader(data), condition)
        expected_starts, expected_loose_ends = reference_starts(data, condition)
        assert set(starts) == set(expected_starts)
        assert set(starts.get_loose_ends()) == set(expected_loose_ends)
        for leaves_only in (True, False):
            subs = subshowers.Subshowers(DataReader(data), leaves_only)
            subs.add_subshower(starts)
            assert len(subs) == len(expected_starts)
            for start, members in subs:
                expected = reference_subshower(data, start, leaves_only)
                assert len(members) == len(expected)
                assert set(members) == set(expected)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_nested_starts(layout):
    data = layout(cascade(3))
    tree = HistoryTree(data)
    # the root, a child of it, a grandchild of that, and a leaf
    root = tree.root
    child = tree.children_of(np.array([root]))[0]
    grandchild = tree.children_of(np.array([child]))[0]
    leaf = np.flatnonzero(tree.is_leaf)[0]
    labels = tree.labels_of(np.array([root, child, grandchild, leaf]))
    for leaves_only in (True, False):
        subs = subshowers.Subshowers(DataReader(data), leaves_only)
        subs.add_subshower(list(labels) + [labels[1]])  # repeats are skipped
        assert len(subs) == 4
        for start, members in subs:
            expected = reference_subshower(data, start, leaves_only)
            assert set(members) == set(expected)
            assert set(subshowers.get_subshower(data, start, leaves_only)) == set(
                expected
            )


def test_tree_on_small_history():
    data = pd.DataFrame(mock_history_dict_2())
    data.at[0, "parent_label"] = -1
    tree = HistoryTree(data)
    assert tree.root == 0
    assert list(tree.parent) == [-1, 0, 0, 1]
    assert list(tree.is_leaf) == [False, False, True, True]
    assert list(tree.subtree_size) == [4, 2, 1, 1]
    assert set(tree.subtree(1)) == {1, 3}
    marked = data["kinetic_energy"].to_numpy() < 8
    assert list(tree.topmost_marked(marked)) == [-1, 1, 2, 1]


def test_self_parent_and_unknown_parent():
    data = pd.DataFrame(mock_history_dict_2())  # the root is its own parent
    data.at[3, "parent_label"] = 99  # a parent that isn't in the history
    tree = HistoryTree(data)
    assert tree.root is None
    assert list(tree.parent) == [-1, 0, 0, -1]
    assert set(tree.subtree(0)) == {0, 1, 2}
    assert set(tree.subtree(3)) == {3}
    assert sorted(tree.preorder) == [0, 1, 2, 3]


def test_save_and_load_round_trip():
    data = cascade(4)
    with tempfile.TemporaryDirectory() as folder:
        shower_folder = os.path.join(folder, "shower")
        fake_history(shower_folder, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subshowers.run(shower_folder, 100.0, leaves_only=True)
        output_path = os.path.join(shower_folder, "subshowers_e100.0.h5")
        output = subshowers.Output.load(output_path)
        # one history read, shared by the starts and the subshowers
        assert output.shower_starts.reader is output.subshowers.reader
        expected_starts, _ = reference_starts(data, subshowers.energy_cut(100.0))
        assert output.subshowers.showerstarts == sorted(expected_starts)
        for start, members in output.subshowers:
            assert set(members) == set(reference_subshower(data, start, True))
        assert output.metadata == {"energy_cut": 100.0, "leaves_only": True}


def test_reader_builds_tree_once():
    data = cascade(5)
    with tempfile.TemporaryDirectory() as folder:
        fake_history(folder, data)
        reader = Reader(folder)
        assert reader.tree is reader.tree
        starts = subshowers.ShowerStarts(reader, subshowers.energy_cut(100.0))
        subs = subshowers.Subshowers(reader)
        subs.add_subshower(starts)
        assert len(subs) == len(starts) > 0
