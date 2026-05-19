import os
import warnings
import pandas as pd
import tempfile
from subshowers import subshowers
from fake_data import mock_history_dict_1, mock_history_dict_2, fake_history


def test_energy_cut():
    cut = subshowers.energy_cut(0.1)
    mock_history_dict = pd.DataFrame.from_dict(mock_history_dict_1())
    assert cut(**(mock_history_dict.iloc[0]))

    mock_history_dict = pd.DataFrame.from_dict(mock_history_dict_2())
    assert not cut(**(mock_history_dict.iloc[0]))
    assert not cut(**(mock_history_dict.iloc[1]))
    assert not cut(**(mock_history_dict.iloc[2]))


class MockReader:
    def __init__(self, history_dict):
        self.data = pd.DataFrame.from_dict(history_dict)
        self.data.at[0, "parent_label"] = -1
        self.folder = "folder.parquet"


def test_ShowerStarts():
    mock_reader1 = MockReader(mock_history_dict_1())
    start_condition = subshowers.energy_cut(-0.1)
    shower_starts = subshowers.ShowerStarts(mock_reader1, start_condition)
    assert len(shower_starts) == 0
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 1
    assert loose_ends[0] == 0
    save_dict = shower_starts._save_dict()
    assert save_dict["folder"] == "folder.parquet"
    assert len(save_dict["starts"]) == 0
    assert len(save_dict["loose_ends"]) == 1
    assert set(save_dict["loose_ends"]) == {0}
    assert save_dict["root"] == 0

    start_condition = subshowers.energy_cut(0.1)
    with warnings.catch_warnings(record=True) as w:
        shower_starts = subshowers.ShowerStarts(mock_reader1, start_condition)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "root particle is also a start particle" in str(w[0].message)
    assert len(shower_starts) == 1
    assert len(shower_starts.get_loose_ends()) == 0

    mock_reader2 = MockReader(mock_history_dict_2())
    start_condition = subshowers.energy_cut(100)
    with warnings.catch_warnings(record=True) as w:
        shower_starts = subshowers.ShowerStarts(mock_reader2, start_condition)
    assert len(shower_starts) == 1
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 0
    save_dict = shower_starts._save_dict()
    assert save_dict["folder"] == "folder.parquet"
    assert len(save_dict["starts"]) == 1
    assert set(save_dict["starts"]) == {0}
    assert len(save_dict["loose_ends"]) == 0
    assert save_dict["root"] == 0

    with tempfile.TemporaryDirectory() as tempdir:
        save_dict["folder"] = tempdir
        fake_history(tempdir, mock_reader2.data)
        copy_shower_starts = subshowers.ShowerStarts._load_dict(save_dict)
        assert len(copy_shower_starts) == 1
        assert len(copy_shower_starts.get_loose_ends()) == 0

    start_condition = subshowers.energy_cut(8)
    shower_starts = subshowers.ShowerStarts(mock_reader2, start_condition)
    assert len(shower_starts) == 2
    set_starts = set(shower_starts)
    assert {1, 2} == set_starts
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 0

    start_condition = subshowers.energy_cut(6)
    with warnings.catch_warnings(record=True) as w:
        shower_starts = subshowers.ShowerStarts(mock_reader2, start_condition)
    assert len(shower_starts) == 1
    set_starts = set(shower_starts)
    assert {1} == set_starts
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 1
    assert loose_ends[0] == 2

    start_condition = subshowers.energy_cut(0.2)
    with warnings.catch_warnings(record=True) as w:
        shower_starts = subshowers.ShowerStarts(mock_reader2, start_condition)
    assert len(shower_starts) == 1
    set_starts = set(shower_starts)
    assert {3} == set_starts
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 1
    assert loose_ends[0] == 2


def test_get_subshowers():
    mock1 = pd.DataFrame.from_dict(mock_history_dict_1())
    with warnings.catch_warnings(record=True) as w:
        subshower = subshowers.get_subshower(mock1, 0, False)
        assert len(w) == 0
    assert set(subshower) == {0}
    subshower = subshowers.get_subshower(mock1, 0, True)
    assert set(subshower) == {0}

    mock2 = pd.DataFrame.from_dict(mock_history_dict_2())
    mock2.at[0, "parent_label"] = -1
    subshower = subshowers.get_subshower(mock2, 0, False)
    assert set(subshower) == {0, 1, 2, 3}
    subshower = subshowers.get_subshower(mock2, 0, True)
    assert set(subshower) == {2, 3}

    subshower = subshowers.get_subshower(mock2, 1, False)
    assert set(subshower) == {1, 3}
    subshower = subshowers.get_subshower(mock2, 1, True)
    assert set(subshower) == {3}

    subshower = subshowers.get_subshower(mock2, 2, False)
    assert set(subshower) == {2}
    subshower = subshowers.get_subshower(mock2, 2, True)
    assert set(subshower) == {2}

    subshower = subshowers.get_subshower(mock2, 3, False)
    assert set(subshower) == {3}
    subshower = subshowers.get_subshower(mock2, 3, True)
    assert set(subshower) == {3}


def test_Subshowers():
    mock_reader = MockReader(mock_history_dict_2())
    subshowers1 = subshowers.Subshowers(mock_reader, True)
    assert len(subshowers1) == 0

    save_dict = subshowers1._save_dict()
    assert save_dict["folder"] == "folder.parquet"
    assert save_dict["starts"] == []
    assert save_dict["subshowers"] == []
    assert save_dict["leaves_only"] == True

    with tempfile.TemporaryDirectory() as tempdir:
        fake_history(tempdir, mock_reader.data)
        save_dict["folder"] = tempdir
        copy_subshowers = subshowers.Subshowers._load_dict(save_dict)
        assert len(copy_subshowers) == 0

    subshowers1.add_subshower([0])
    assert len(subshowers1) == 1
    assert len(subshowers1.showerstarts) == 1
    subshowers1.add_subshower([1])
    assert len(subshowers1) == 2
    assert len(subshowers1.showerstarts) == 2
    subshowers1.add_subshower([0])
    assert len(subshowers1) == 2
    assert len(subshowers1.showerstarts) == 2

    save_dict = subshowers1._save_dict()
    assert save_dict["folder"] == "folder.parquet"
    assert sorted(save_dict["starts"]) == [0, 0, 1]
    assert sorted(save_dict["subshowers"]) == [2, 3, 3]
    assert save_dict["leaves_only"] == True

    with tempfile.TemporaryDirectory() as tempdir:
        save_dict["folder"] = tempdir
        fake_history(tempdir, mock_reader.data)
        copy_subshowers = subshowers.Subshowers._load_dict(save_dict)
    assert len(copy_subshowers) == 2
    assert len(copy_subshowers.showerstarts) == 2
    assert set(copy_subshowers[0][1]) == {2, 3}
    assert set(copy_subshowers[1][1]) == {3}

    start, shower = subshowers1[0]
    assert start == 0
    assert set(shower) == {2, 3}

    start, shower = subshowers1[1]
    assert start == 1
    assert set(shower) == {3}

    for start, shower in subshowers1:
        if start == 0:
            assert set(shower) == {2, 3}
        elif start == 1:
            assert set(shower) == {3}
        else:
            assert False


def test_Output():
    with tempfile.TemporaryDirectory() as tempdir:
        mock_reader = MockReader(mock_history_dict_2())
        mock_reader.folder = tempdir

        start_condition = subshowers.energy_cut(100)
        with warnings.catch_warnings(record=True) as w:
            shower_starts = subshowers.ShowerStarts(mock_reader, start_condition)
            assert len(w) == 1
        subshowers1 = subshowers.Subshowers(mock_reader, True)
        subshowers1.add_subshower([0, 1])

        output = subshowers.Output(shower_starts, subshowers1, test="test")
        assert output.shower_starts == shower_starts
        assert output.subshowers == subshowers1

        fake_history(tempdir, mock_reader.data)
        file_name = os.path.join(tempdir, "output.h5")
        output.save(file_name)
        output2 = subshowers.Output.load(file_name)
        new_shower_starts = output2.shower_starts
        new_subshowers = output2.subshowers
        metadata = output2.metadata

    assert metadata["test"] == "test"

    # check shower starts saved and loaded correctly
    assert len(new_shower_starts) == 1
    loose_ends = new_shower_starts.get_loose_ends()
    assert len(loose_ends) == 0
    save_dict = new_shower_starts._save_dict()
    assert save_dict["folder"] == tempdir
    assert len(save_dict["starts"]) == 1
    assert set(save_dict["starts"]) == {0}
    assert len(save_dict["loose_ends"]) == 0
    assert save_dict["root"] == 0

    # check subshowers saved and loaded correctly
    assert len(new_subshowers) == 2
    assert len(new_subshowers.showerstarts) == 2
    save_dict = new_subshowers._save_dict()
    assert save_dict["folder"] == tempdir
    assert sorted(save_dict["starts"]) == [0, 0, 1]
    assert sorted(save_dict["subshowers"]) == [2, 3, 3]
    assert save_dict["leaves_only"] == True


if __name__ == "__main__":
    test_energy_cut()
    test_ShowerStarts()
    test_get_subshowers()
    test_Subshowers()
    test_Output()
