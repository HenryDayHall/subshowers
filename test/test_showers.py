import tempfile
from subshowers import showers
from fake_data import mock_history_dict_1, mock_history_dict_2, fake_history


def test_reader():
    with tempfile.TemporaryDirectory() as tempdir:
        history_dict = mock_history_dict_1()
        fake_history(tempdir, history_dict)
        reader = showers.Reader(tempdir)
        assert len(reader) == 1
        for col in showers.Reader.history_columns:
            assert col in reader.data
            if col != "parent_label":
                assert reader.data.at[0, col] == 0
            else:
                assert reader.data.at[0, col] == -1

        history_dict = mock_history_dict_2()
        fake_history(tempdir, history_dict)
        reader = showers.Reader(tempdir)
        assert len(reader) == 4
