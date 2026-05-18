import pandas as _pd
import yaml as _yaml

_types_to_yaml = [int, float, str, bool, type(None)]


def allocate_to_yaml(item):
    for t in _types_to_yaml:
        if isinstance(item, t):
            return (True, t(item))
    if isinstance(item, list):
        is_list_of_strings = all(isinstance(i, str) for i in item)
        if is_list_of_strings:
            converted = [str(i) for i in item]
        else:
            converted = None
        return (is_list_of_strings, converted)
    return (False, None)


def save(path_base: str, *regular_arrays, **irregular_items):
    """
    Save data to disk.

    Parameters
    ---------
    path : str
        Path to save data to, without file extension.
    regular_arrays : (dict of arrays) or pd.DataFrame
        Regular arrays of data to save,
        each dictionary must contain arrays of the same length,
        with at most 2 dimensions.
        Each will be written to a pandas h5 dataset.
    irregular_items : array or basic type or list of strings
        Any other data to save.
        If it is in _types_to_yaml, or a list of strings,
        it will be written to a yaml file,
        otherwise it will be stored as a seperate h5 dataset by pandas.
    """
    h5_path = path_base + ".h5"
    used_h5_names = []
    for i, item in enumerate(regular_arrays):
        key = f"regular_array_{i}"
        if isinstance(item, _pd.DataFrame):
            item.to_hdf(h5_path, key)
            new_cols = [key]
        elif isinstance(item, dict):
            new_cols = _save_regular_array(h5_path, key, item)
        else:
            raise ValueError(
                "Regular arrays must be either a pandas DataFrame or a dictionary."
            )
        for col in new_cols:
            if col in used_h5_names:
                raise ValueError(f"Tried to save duplicate column: {col}")
            used_h5_names.append(col)
    yaml_dict = {}
    for key, value in irregular_items.items():
        should_yaml, yaml_value = allocate_to_yaml(value)
        if should_yaml:
            # no need to check if the key is in use
            # all keys for the yaml come from the kwargs
            yaml_dict[key] = yaml_value
        else:
            if key in used_h5_names:
                raise ValueError(f"Tried to save duplicate column: {key}")
            used_h5_names.append(key)
            value = _pd.DataFrame(value)
            value.to_hdf(h5_path, key)
    _yaml.dump(yaml_dict, open(path_base + ".yaml", "w"))


def _save_regular_array(h5_path, key, regular_array):
    order = sorted(regular_array.keys())
    col_names = []
    values = []
    for key in order:
        value = regular_array[key]
        shape = value.shape
        if len(shape) == 1:
            col_names.append(key)
            values.append(value)
        elif len(shape) == 2:
            for i in range(shape[1]):
                col_names.append(key + "_columnNumber_" + str(i))
                values.append(value[:, i])
        else:
            raise ValueError("Regular arrays must have 1 or 2 dimensions.")

    df = _pd.DataFrame(values, columns=col_names)
    df.to_hdf(h5_path, key)
    return col_names


def load(path_base: str):
    found_items = {}
    found_2d_component = {}
    h5_path = path_base + ".h5"
    h5_keys = list(_pd.HDFStore(h5_path).keys())
    h5_values = [_pd.read_hdf(h5_path, key) for key in h5_keys]
    for key, value in zip(h5_keys, h5_values):
        if "_columnNumber_" in key:
            base_key, column_number = key.split("_columnNumber_")
            if base_key not in found_2d_component:
                found_2d_component[base_key] = {column_number: value}
            else:
                found_2d_component[base_key][column_number] = value
        else:
            found_items[key] = value

    for key, columns in found_2d_component.items():
        n_columns = len(value)
        column_names = [str(i) for i in range(n_columns)]
        df = _pd.DataFrame(data=columns, index=column_names)
        found_items[key] = df

    yaml_dict = _yaml.safe_load(open(path_base + ".yaml", "r"))
    for key, value in yaml_dict.items():
        found_items[key] = value

    return found_items
