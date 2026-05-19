import pandas as _pd
import yaml as _yaml

_types_to_yaml = [bool, int, float, str, type(None)]


def allocate_to_yaml(item):
    if item is None:
        return (True, None)
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


def get_h5_keys(h5_path):
    with _pd.HDFStore(h5_path) as h5:
        return [key.lstrip("/") for key in h5.keys()]


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
    if path_base.endswith(".h5"):
        path_base = path_base[:-3]
    h5_path = path_base + ".h5"
    prior_keys = set(load(h5_path).keys())
    used_h5_names = get_h5_keys(h5_path)
    i = 0
    h5_key = f"regular_array_{i}"
    for item in regular_arrays:
        while h5_key in used_h5_names:
            i += 1
            h5_key = f"regular_array_{i}"
        if isinstance(item, _pd.DataFrame):
            columns = set(item.columns)
            item.to_hdf(h5_path, key=h5_key, mode="a")
        elif isinstance(item, dict):
            columns = set(item.keys())
            _save_regular_array(h5_path, h5_key, item)
        else:
            raise ValueError(
                "Regular arrays must be either a pandas DataFrame or a dictionary."
            )
        if prior_keys.intersection(columns):
            raise ValueError(f"Tried to save duplicate column: {columns}")
        prior_keys.update(columns)
        used_h5_names.append(h5_key)
                
    yaml_dict = {}
    for key, value in irregular_items.items():
        if key in prior_keys:
            raise ValueError(f"Tried to save duplicate column: {key}")
        prior_keys.add(key)
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
            value.to_hdf(h5_path, key=key, mode="a")
    _yaml.dump(yaml_dict, open(path_base + ".yaml", "w"))


def _save_regular_array(h5_path, key, regular_array):
    order = sorted(regular_array.keys())
    values = {}
    for column in order:
        value = regular_array[column]
        shape = value.shape
        if len(shape) == 1:
            values[column] = value
        elif len(shape) == 2:
            for i in range(shape[1]):
                col_name = column + "_columnNumber_" + str(i)
                values[col_name] = value[:, i]
        else:
            raise ValueError("Regular arrays must have 1 or 2 dimensions.")

    df = _pd.DataFrame(data=values)
    df.to_hdf(h5_path, key=key, mode="a")
    return df.columns


def load(path_base: str):
    if path_base.endswith(".h5"):
        path_base = path_base[:-3]
    found_items = {}
    found_2d_component = {}
    h5_path = path_base + ".h5"
    h5_keys = get_h5_keys(h5_path)
    h5_values = [_pd.read_hdf(h5_path, key, mode="r") for key in h5_keys]
    for key, value in zip(h5_keys, h5_values):
        if key.startswith("regular_array"):
            for column, col_value in value.items():
                if "_columnNumber_" in column:
                    base_key, column_number = column.split("_columnNumber_")
                    if base_key not in found_2d_component:
                        found_2d_component[base_key] = {column_number: col_value}
                    else:
                        found_2d_component[base_key][column_number] = col_value
                else:
                    found_items[column] = col_value
        else:
            found_items[key] = value

    for key, columns in found_2d_component.items():
        df = _pd.DataFrame(data=columns)
        found_items[key] = df

    try:
        yaml_dict = _yaml.safe_load(open(path_base + ".yaml", "r"))
    except FileNotFoundError:
        yaml_dict = {}
    for key, value in yaml_dict.items():
        found_items[key] = value

    return found_items
