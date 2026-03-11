import os
import pandas as pd
import tempfile
import subshowers


def mock_history_dict_1():
    """Simplist possible history dict"""
    history_dict = {
        "shower": [0],
        "pdg": [0],
        "px": [0],
        "py": [0],
        "pz": [0],
        "x": [0],
        "y": [0],
        "z": [0],
        "kinetic_energy": [0],
        "time": [0],
        "label": [0],
        "parent_label": [-1],
    }
    return history_dict


def mock_history_dict_2():
    """ History dict with a small tree of
    parent -> child child -> grandchild
    """
    history_dict = {
        "shower": [0, 0, 0, 0],
        "pdg": [0, 1, 2, 3],
        "px": [0, 0.5, 0, 0],
        "py": [1, 0, 0.5, 0],
        "pz": [2, 0, 0, 0.5],
        "x": [0, 2, 0, 0],
        "y": [0, 1.1, 0, 0],
        "z": [0, 0, 0.1, 0],
        "kinetic_energy": [10, 5, 7, 0.1],
        "time": [0.1, 0, 0.5, 0],
        "label": [0, 1, 2, 3],
        "parent_label": [0, 0, 0, 1],
    }
    return history_dict


def test_reader():
    with tempfile.TemporaryDirectory() as tempdir:
        hist_dir = os.path.join(tempdir, "history")
        os.mkdir(hist_dir)
        history_dict = mock_history_dict_1()
        history_path = os.path.join(hist_dir, "history.parquet")
        pd.DataFrame(history_dict).to_parquet(history_path)
        reader = subshowers.Reader(tempdir)
        assert len(reader) == 1
        for col in subshowers.Reader.history_columns:
            assert col in reader.data
            if col != "parent_label":
                assert reader.data.at[0, col] == 0
            else:
                assert reader.data.at[0, col] == -1

        history_dict = mock_history_dict_2()
        pd.DataFrame(history_dict).to_parquet(history_path)
        reader = subshowers.Reader(tempdir)
        assert len(reader) == 4


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
    shower_starts = subshowers.ShowerStarts(mock_reader1, start_condition)
    assert len(shower_starts) == 1
    assert len(shower_starts.get_loose_ends()) == 0

    mock_reader2 = MockReader(mock_history_dict_2())
    start_condition = subshowers.energy_cut(100)
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
        history_folder = os.path.join(tempdir, "history")
        os.mkdir(history_folder)
        history_path = os.path.join(history_folder, "history.parquet")
        mock_reader2.data.to_parquet(history_path)
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
    shower_starts = subshowers.ShowerStarts(mock_reader2, start_condition)
    assert len(shower_starts) == 1
    set_starts = set(shower_starts)
    assert {1} == set_starts
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 1
    assert loose_ends[0] == 2

    start_condition = subshowers.energy_cut(0.2)
    shower_starts = subshowers.ShowerStarts(mock_reader2, start_condition)
    assert len(shower_starts) == 1
    set_starts = set(shower_starts)
    assert {3} == set_starts
    loose_ends = shower_starts.get_loose_ends()
    assert len(loose_ends) == 1
    assert loose_ends[0] == 2


def test_get_subshowers():
    mock1 = pd.DataFrame.from_dict(mock_history_dict_1())
    subshower = subshowers.get_subshower(mock1, 0, False)
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
        save_dict["folder"] = tempdir
        history_folder = os.path.join(tempdir, "history")
        os.mkdir(history_folder)
        history_path = os.path.join(history_folder, "history.parquet")
        mock_reader.data.to_parquet(history_path)
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
        history_folder = os.path.join(tempdir, "history")
        os.mkdir(history_folder)
        history_path = os.path.join(history_folder, "history.parquet")
        mock_reader.data.to_parquet(history_path)
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
    mock_reader = MockReader(mock_history_dict_2())
    start_condition = subshowers.energy_cut(100)
    shower_starts = subshowers.ShowerStarts(mock_reader, start_condition)
    subshowers1 = subshowers.Subshowers(mock_reader, True)
    subshowers1.add_subshower([0, 1])

    output = subshowers.Output(shower_starts, subshowers1, test="test")
    assert output.shower_starts == shower_starts
    assert output.subshowers == subshowers1
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        output.save(f.name)
        output2 = subshowers.Output.load(f.name)
        new_shower_starts = output2.shower_starts
        new_subshowers = output2.subshowers
        metadata = output2.metadata

    assert metadata["test"] == "test"

    # check shower starts saved and loaded correctly
    assert len(new_shower_starts) == 1
    loose_ends = new_shower_starts.get_loose_ends()
    assert len(loose_ends) == 0
    save_dict = new_shower_starts._save_dict()
    assert save_dict["folder"] == "folder.parquet"
    assert len(save_dict["starts"]) == 1
    assert set(save_dict["starts"]) == {0}
    assert len(save_dict["loose_ends"]) == 0
    assert save_dict["root"] == 0

    # check subshowers saved and loaded correctly
    assert len(new_subshowers) == 2
    assert len(new_subshowers.showerstarts) == 2
    save_dict = new_subshowers._save_dict()
    assert save_dict["folder"] == "folder.parquet"
    assert sorted(save_dict["starts"]) == [0, 0, 1]
    assert sorted(save_dict["subshowers"]) == [2, 3, 3]
    assert save_dict["leaves_only"] == True
