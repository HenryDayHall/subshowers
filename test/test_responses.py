"""
Tests for subshowers.responses.Responses, and the observables worked out from
it.
"""

import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from subshowers import detector, observables, subshowers
from subshowers.responses import Responses
from subshowers.sampling import DatasetSource, SubshowerSampler
from fake_data import fake_history, grow_cascade


def made_up_sample(rng, n_detectors, source="model", hits=3):
    """A subshower as a generative model might give it: no leaves known."""
    energy = np.zeros(n_detectors)
    hit = rng.choice(n_detectors, size=hits, replace=False)
    energy[hit] = rng.uniform(0.1, 2.0, hits)
    return SimpleNamespace(
        source=source,
        start_pdg=int(rng.choice([22, 11, -11])),
        start_kinetic_energy=float(rng.uniform(1, 100)),
        start_time=float(rng.uniform()),
        start_position=rng.normal(size=3),
        start_direction=rng.normal(size=3),
        energy_per_detector=energy,
    )


def test_from_samples():
    rng = np.random.default_rng(1)
    samples = [made_up_sample(rng, 20, hits=hits) for hits in [0, 3, 1, 0, 5]]
    responses = Responses.from_samples(samples, 20)
    assert len(responses) == 5
    assert responses.n_hit_detectors.tolist() == [0, 3, 1, 0, 5]
    assert responses.detected.tolist() == [False, True, True, False, True]
    assert np.allclose(
        responses.total_detected_energy,
        [sample.energy_per_detector.sum() for sample in samples],
    )
    for i, sample in enumerate(samples):
        assert np.array_equal(
            responses.energy_per_detector(i), sample.energy_per_detector
        )
        stored = responses.sample(i)
        assert stored.source == "model"
        assert stored.start_pdg == sample.start_pdg
        assert np.array_equal(stored.start_position, sample.start_position)
        assert stored.total_detected_energy == pytest.approx(
            sample.energy_per_detector.sum()
        )
        # a made up sample has no leaves to count
        assert stored.n_leaves == -1
        assert np.isnan(stored.total_leaf_energy)
    named = Responses.from_samples(samples, 20, source="flow model")
    assert set(named.values["source"]) == {"flow model"}


def test_select_concatenate_save_load(tmp_path):
    rng = np.random.default_rng(2)
    first = Responses.from_samples(
        [made_up_sample(rng, 10, "a", hits) for hits in [2, 0, 4]],
        10,
        metadata={"made": "here"},
    )
    second = Responses.from_samples(
        [made_up_sample(rng, 10, "b", hits) for hits in [1, 3]], 10
    )
    both = Responses.concatenate([first, second])
    assert both.values["source"].tolist() == ["a", "a", "a", "b", "b"]
    for i in range(3):
        assert np.array_equal(both.energy_per_detector(i), first.energy_per_detector(i))
    for i in range(2):
        assert np.array_equal(
            both.energy_per_detector(3 + i), second.energy_per_detector(i)
        )

    detected = both.select(both.detected)
    assert len(detected) == 4
    for i, j in enumerate(np.flatnonzero(both.detected)):
        assert np.array_equal(
            detected.energy_per_detector(i), both.energy_per_detector(j)
        )
        assert detected.values["start_time"][i] == both.values["start_time"][j]
    picked = both.select([4, 0])
    assert np.array_equal(picked.energy_per_detector(0), both.energy_per_detector(4))

    path = str(tmp_path / "responses.npz")
    first.save(path)
    assert Responses.read_metadata(path) == {"made": "here"}
    loaded = Responses.load(path)
    assert loaded.metadata == {"made": "here"}
    assert loaded.n_detectors == 10
    for name in first.values:
        np.testing.assert_array_equal(loaded.values[name], first.values[name])
    for i in range(len(first)):
        assert np.array_equal(
            loaded.energy_per_detector(i), first.energy_per_detector(i)
        )

    Responses.from_samples([], 10).save(path)
    assert len(Responses.load(path)) == 0


def test_detected_leaves(tmp_path):
    rng = np.random.default_rng(4)
    known = [made_up_sample(rng, 10, "a", hits) for hits in [2, 0, 3]]
    for sample, leaves in zip(known, [[5, 9], [], [1, 2, 3]]):
        sample.detected_leaf_indices = np.array(leaves)
    unknown = [made_up_sample(rng, 10, "b", hits) for hits in [1, 2]]
    with_leaves = Responses.from_samples(known, 10)
    assert with_leaves.keeps_detected_leaves
    assert with_leaves.detected_leaf_indices(0).tolist() == [5, 9]
    assert with_leaves.detected_leaf_indices(1).tolist() == []
    without = Responses.from_samples(unknown, 10)
    assert not without.keeps_detected_leaves
    assert without.detected_leaf_indices(0) is None

    both = Responses.concatenate([with_leaves, without])
    assert both.detected_known.tolist() == [True, True, True, False, False]
    assert both.detected_leaf_indices(2).tolist() == [1, 2, 3]
    picked = both.select([2, 4, 0])
    assert picked.detected_leaf_indices(0).tolist() == [1, 2, 3]
    assert picked.detected_leaf_indices(1) is None
    assert picked.detected_leaf_indices(2).tolist() == [5, 9]
    # they go through StoredSamples, and back
    again = Responses.from_samples(picked.samples(), 10)
    np.testing.assert_array_equal(again.detected_known, picked.detected_known)
    np.testing.assert_array_equal(again.detected_leaves, picked.detected_leaves)

    path = str(tmp_path / "responses.npz")
    both.save(path)
    loaded = Responses.load(path)
    np.testing.assert_array_equal(loaded.detected_offsets, both.detected_offsets)
    np.testing.assert_array_equal(loaded.detected_leaves, both.detected_leaves)
    np.testing.assert_array_equal(loaded.detected_known, both.detected_known)

    with pytest.raises(ValueError, match="aren't known"):
        Responses(
            10,
            without.values,
            without.hit_offsets,
            without.hit_detectors,
            without.hit_energies,
            detected_offsets=[0, 1, 1],
            detected_leaves=[7],
            detected_known=[False, False],
        )
    with pytest.raises(ValueError, match="detected_offsets"):
        Responses(
            10,
            without.values,
            without.hit_offsets,
            without.hit_detectors,
            without.hit_energies,
            detected_offsets=[0, 2],
            detected_leaves=[7, 8],
        )


def test_load_version_1(tmp_path):
    """Responses saved before detected leaves were kept still load, without them."""
    rng = np.random.default_rng(5)
    responses = Responses.from_samples([made_up_sample(rng, 10) for _ in range(3)], 10)
    path = str(tmp_path / "old.npz")
    arrays = {
        "format_version": np.array(1),
        "n_detectors": np.array(10),
        "metadata": np.array('{"made": "before"}'),
        "hit_offsets": responses.hit_offsets,
        "hit_detectors": responses.hit_detectors,
        "hit_energies": responses.hit_energies,
    }
    for name, column in responses.values.items():
        arrays[f"value_{name}"] = column
    np.savez_compressed(path, **arrays)
    loaded = Responses.load(path)
    assert len(loaded) == 3 and loaded.metadata == {"made": "before"}
    assert not loaded.keeps_detected_leaves
    for i in range(3):
        np.testing.assert_array_equal(
            loaded.energy_per_detector(i), responses.energy_per_detector(i)
        )
    arrays["format_version"] = np.array(7)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="format version 7"):
        Responses.load(path)


def test_bad_responses():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="shape"):
        Responses.from_samples([made_up_sample(rng, 10)], 11)
    ten = Responses.from_samples([made_up_sample(rng, 10)], 10)
    twelve = Responses.from_samples([made_up_sample(rng, 12)], 12)
    with pytest.raises(ValueError, match="different arrays"):
        Responses.concatenate([ten, twelve])
    with pytest.raises(ValueError, match="detectors"):
        observables.record_of(ten, detector.generate_pierre_auger("sd750"))


def test_dataset_samples(tmp_path):
    """Responses keep all that observables need of sampled subshowers."""
    folder = str(tmp_path / "e1e5_0")
    fake_history(folder, grow_cascade(np.random.default_rng(1), 4.0e3, 170_000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        subshowers.run(folder, 100.0, leaves_only=True)
    array = detector.generate_at_ground(300, 6000.0, 300.0, 20.0, seed=0)
    source = DatasetSource(
        folder, output_path=os.path.join(folder, "subshowers_e100.0.h5"), name="e1e5_0"
    )
    samples = list(SubshowerSampler([source], array, seed=1).iter_epoch())
    assert any(sample.n_hit_detectors for sample in samples)

    responses = Responses.from_samples(samples, len(array))
    assert set(responses.values["source"]) == {"e1e5_0"}
    for name in ["subshower_number", "n_leaves", "n_detected_leaves", "start_pdg"]:
        assert responses.values[name].tolist() == [
            getattr(sample, name) for sample in samples
        ]
    assert responses.keeps_detected_leaves
    for i, sample in enumerate(samples):
        np.testing.assert_array_equal(
            responses.detected_leaf_indices(i), sample.detected_leaf_indices
        )
        assert len(sample.detected_leaf_indices) == sample.n_detected_leaves
    record = observables.record_of(responses, array)
    expected = observables.new_record()
    for sample in samples:
        observables.add_sample(expected, sample, array)
    for key, values in expected.items():
        assert len(record[key]) == len(values)
        for value, stored in zip(values, record[key]):
            np.testing.assert_array_equal(
                np.asarray(stored, dtype=float), np.asarray(value, dtype=float)
            )
