import os
import pandas as pd
import numpy as np
import tempfile
from subshowers import raw_save_load


def test_allocate_to_yaml():
    should_allocate, converted = raw_save_load.allocate_to_yaml(1)
    assert should_allocate
    assert converted == 1

    should_allocate, converted = raw_save_load.allocate_to_yaml(1.0)
    assert should_allocate
    assert converted == 1.0

    should_allocate, converted = raw_save_load.allocate_to_yaml("1")
    assert should_allocate
    assert converted == "1"

    should_allocate, converted = raw_save_load.allocate_to_yaml(True)
    assert should_allocate
    assert isinstance(converted, bool)
    assert converted

    should_allocate, converted = raw_save_load.allocate_to_yaml([1, 2, 3])
    assert not should_allocate

    should_allocate, converted = raw_save_load.allocate_to_yaml(None)
    assert should_allocate
    assert converted is None

    should_allocate, converted = raw_save_load.allocate_to_yaml(["1", "2", "3"])
    assert should_allocate
    for i in range(3):
        assert converted[i] == str(i + 1)


def test_save_load():
    """
    Way easier to test both methods together
    """
    # shouldn't choke on empty content
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        raw_save_load.save(file_name)

    # a singular regular array
    regular_1 = pd.DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        raw_save_load.save(file_name, regular_1)
        loaded = raw_save_load.load(file_name)
        for key in regular_1.columns:
            assert key in loaded
            assert np.all(regular_1[key].values == loaded[key].values)

    # a dictionary of regular arrays
    regular_2 = {"c": np.zeros((2, 3)), "d": np.ones((2, 3))}
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        raw_save_load.save(file_name, regular_2)
        loaded = raw_save_load.load(file_name)
        for key in regular_2.keys():
            assert key in loaded
            assert np.all(regular_2[key] == loaded[key])

    # should be able to save both one at a time
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        raw_save_load.save(file_name, regular_1)
        # loading in between should make no difference
        loaded = raw_save_load.load(file_name)
        raw_save_load.save(file_name, regular_2)
        loaded = raw_save_load.load(file_name)
        for key in regular_1.columns:
            assert key in loaded
            assert np.all(regular_1[key].values == loaded[key].values)
        for key in regular_2.keys():
            assert key in loaded
            assert np.all(regular_2[key] == loaded[key])

    # should be able to save both in one call
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        raw_save_load.save(file_name, regular_1, regular_2)
        loaded = raw_save_load.load(file_name)
        for key in regular_1.columns:
            assert key in loaded
            assert np.all(regular_1[key].values == loaded[key].values)
        for key in regular_2.keys():
            assert key in loaded
            assert np.all(regular_2[key] == loaded[key])

    # now we add some named irregular stuff
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        rand_array = np.random.randint(0, 100, (20,))
        raw_save_load.save(file_name, a=1, b=True, c=["Hi", "there"], d=rand_array)
        loaded = raw_save_load.load(file_name)
        assert "a" in loaded
        assert "b" in loaded
        assert "c" in loaded
        assert "d" in loaded
        assert loaded["a"] == 1
        assert isinstance(loaded["b"], bool)
        assert loaded["b"]
        for i, word in enumerate(["Hi", "there"]):
            assert loaded["c"][i] == word
        assert np.all(rand_array == loaded["d"].to_numpy().flatten())

    # Can also do both together
    with tempfile.TemporaryDirectory() as d:
        file_name = os.path.join(d, "test.h5")
        raw_save_load.save(
            file_name,
            regular_1,
            regular_2,
            a_x=1,
            b_x=True,
            c_x=["Hi", "there"],
            d_x=rand_array,
        )
        loaded = raw_save_load.load(file_name)
        assert "a" in loaded
        assert "b" in loaded
        assert "c" in loaded
        assert "d" in loaded
        assert "a_x" in loaded
        assert "b_x" in loaded
        assert "c_x" in loaded
        assert "d_x" in loaded

        assert loaded["a_x"] == 1
        assert isinstance(loaded["b_x"], bool)
        assert loaded["b_x"]
        for i, word in enumerate(["Hi", "there"]):
            assert loaded["c_x"][i] == word
        assert np.all(rand_array == loaded["d_x"].to_numpy().flatten())

        for key in regular_1.columns:
            assert key in loaded
            assert np.all(regular_1[key].values == loaded[key].values)
        for key in regular_2.keys():
            assert key in loaded
            assert np.all(regular_2[key] == loaded[key])


if __name__ == "__main__":
    test_allocate_to_yaml()
    test_save_load()
