import os as _os
import sys as _sys
import pandas as _pd
import plotly.graph_objects as _go
import numpy as _np
from plotly.express.colors import sample_colorscale
import matplotlib as _matplotlib
import matplotlib.pyplot as _plt

import skimage.draw as _skdraw
from subshowers import detector


def add_shower_skeleton(
    fig,
    reader,
    max_points=1000,
    x_axis="x",
    y_axis="z",
    colour="kinetic_energy",
    name="Shower Skeleton",
    log_colour=True,
):
    total_particles = len(reader)
    n_points = min(total_particles, max_points)
    multiplier = total_particles / n_points
    point_idxs = [int(i * multiplier) for i in range(n_points)]

    if colour in reader.history_columns:
        colour = reader.data[colour][point_idxs]

    if log_colour:
        log_positions = [0.0, 0.0000001, 0.000001, 0.00001, 0.0001, 1.0]
        n_positions = len(log_positions)
        samples = sample_colorscale("Viridis", n_positions)
        colour_scale = [
            [log_positions[i], samples[i][:-1] + ", 0.1)"] for i in range(n_positions)
        ]
    else:
        colour_scale = None

    fig.add_trace(
        _go.Scatter(
            x=reader.data[x_axis][point_idxs],
            y=reader.data[y_axis][point_idxs],
            marker_color=colour,
            marker_colorscale=colour_scale,
            mode="markers",
            name=name,
            marker_showscale=True,
            marker_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"),
        )
    )
    return fig


class Overview:
    def __init__(
        self, reader, x_axis="x", y_axis="z", grid_size=500, force_recalculate=False
    ):
        self.reader = reader
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.grid_size = grid_size

        # set up the grid
        min_x = _np.min(reader.data[x_axis]) - 1
        max_x = _np.max(reader.data[x_axis]) + 1
        min_y = min(0, _np.min(reader.data[y_axis])) - 1
        max_y = _np.max(reader.data[y_axis]) + 1

        self.x_bins = _np.linspace(min_x, max_x, grid_size + 1)
        self.y_bins = _np.linspace(min_y, max_y, grid_size + 1)

        self.grid = _np.zeros((grid_size, grid_size))
        if force_recalculate:
            self.relations_to_hist()
        else:
            try:
                self._load()
            except FileNotFoundError:
                self.relations_to_hist()

    def relations_to_hist(self):
        total_particles = len(self.reader)
        inverse_total = 1.0 / total_particles

        binned_x = _np.digitize(self.reader.data[self.x_axis], self.x_bins) - 1
        binned_y = _np.digitize(self.reader.data[self.y_axis], self.y_bins) - 1

        label_to_index = {
            label: idx for idx, label in enumerate(self.reader.data["label"])
        }
        label_to_index[-1] = -1
        parent_index = self.reader.data["parent_label"].map(label_to_index)

        for child, parent in enumerate(parent_index):
            if parent == -1:
                continue
            if child % 1000 == 0:
                print(f"{child*inverse_total:00.0%}", end="\r")

            rows, cols = _skdraw.line(
                binned_x[child], binned_y[child], binned_x[parent], binned_y[parent]
            )
            self.grid[cols, rows] = 1
        print()
        self._save()

    def _save(self):
        path = self._save_path()
        metadata = _pd.DataFrame.from_dict(
            {
                "x_axis": [self.x_axis],
                "y_axis": [self.y_axis],
                "grid_size": [self.grid_size],
                "reader_folder": [self.reader.folder],
            }
        )
        metadata.to_hdf(path, key="metadata", mode="a")
        bins = _pd.DataFrame.from_dict(
            {
                "x_bins": self.x_bins,
                "y_bins": self.y_bins,
            }
        )
        bins.to_hdf(path, key="bins", mode="a")
        grid = _pd.DataFrame(self.grid)
        grid.to_hdf(path, key="grid", mode="a")

    def _load(self):
        path = self._save_path()
        metadata = _pd.read_hdf(path, key="metadata")
        self.x_axis = metadata["x_axis"].values[0]
        self.y_axis = metadata["y_axis"].values[0]
        self.grid_size = metadata["grid_size"].values[0]
        self.reader.folder = metadata["reader_folder"].values[0]
        bins = _pd.read_hdf(path, key="bins")
        self.x_bins = bins["x_bins"].values
        self.y_bins = bins["y_bins"].values
        grid = _pd.read_hdf(path, key="grid")
        self.grid = grid.values

    def _save_path(self):
        reader_path = self.reader.folder
        save_path = _os.path.join(reader_path, "plotting_overview.h5")
        return save_path


def add_shower_relations(
    fig,
    overview,
    name="Shower Relations",
):
    # make a white to gray colorscale
    colorscale = [
        [0.0, "rgb(0,0,0)"],
        [1.0, "rgb(155,155,155)"],
    ]

    fig.add_trace(
        _go.Heatmap(
            z=overview.grid,
            x=overview.x_bins,
            y=overview.y_bins,
            colorscale=colorscale,
            colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"),
            name=name,
        )
    )
    return fig


def add_subshower(
    fig,
    subshowers,
    reader,
    start_index,
    max_points=1000,
    x_axis="x",
    y_axis="z",
    colour="blue",
    name="Subshower",
):
    subshower_number = subshowers.showerstarts.index(start_index)
    subshower = subshowers.subshowers[subshower_number]
    total_particles = len(subshower)
    n_points = min(total_particles, max_points)
    multiplier = total_particles / n_points
    point_idxs = [subshower[int(i * multiplier)] for i in range(n_points)]
    point_idxs.append(start_index)

    size = [4] * n_points + [10]

    if colour in reader.history_columns:
        colour = reader.data[colour][point_idxs]

    fig.add_trace(
        _go.Scatter(
            x=reader.data[x_axis][point_idxs],
            y=reader.data[y_axis][point_idxs],
            marker_color=colour,
            marker_line_color=colour,
            marker_symbol="x-thin",
            mode="markers",
            marker_size=size,
            name=name,
            visible="legendonly",
        )
    )

    return fig


def add_selected_subshowers(
    fig,
    subshowers,
    reader,
    max_subshowers=10,
    max_points=1000,
    x_axis="x",
    y_axis="z",
):
    colours = sample_colorscale("jet", max_subshowers)
    n_subshowers = min(len(subshowers), max_subshowers)
    chosen = _np.random.choice(len(subshowers), size=n_subshowers, replace=False)
    kin_es = []

    for i in chosen:
        subshower = subshowers[i]
        start_index = subshower[0]
        start_kin_e = reader.data["kinetic_energy"][start_index]
        kin_es.append(start_kin_e)
    order = _np.argsort(kin_es)

    chosen = _np.array(chosen)[order]
    kin_es = _np.array(kin_es)[order]

    for i, kine, colour in zip(chosen, kin_es, colours):
        name = f"Subshower, KE = {kine:.2e}"
        subshower = subshowers[i]
        start_index = subshower[0]
        fig = add_subshower(
            fig,
            subshowers,
            reader,
            start_index,
            max_points=max_points,
            x_axis=x_axis,
            y_axis=y_axis,
            colour=colour,
            name=name,
        )
    return fig


def add_scatter(sensitive_regions, extent, energy_per_region=None, axis=None, **kwargs):
    xs, ys = _np.copy(sensitive_regions.centers[:, 0]), _np.copy(
        sensitive_regions.centers[:, 1]
    )

    mask = (xs < extent[1]) * (xs > extent[0]) * (ys < extent[3]) * (ys > extent[2])
    if not _np.any(mask):
        return None
    if energy_per_region is None:
        c = None
    else:
        c = energy_per_region[mask]
        print(f"Total colour {c.sum()}")
        if _np.all(c == 0):
            c[:] = 1
        elif _np.any(c == 0):
            c[c == 0] = 0.01 * _np.min(c[c > 0])
        kwargs["norm"] = _matplotlib.colors.LogNorm()

    if axis is None:
        axis = _plt.gca()

    path_collection = axis.scatter(xs[mask], ys[mask], c=c, **kwargs)
    return path_collection


def calculate_extent(grid_center, grid_size):
    extent = _np.full(4, grid_size / 2)
    extent[[0, 2]] *= -1
    extent[[0, 1]] += grid_center[0]
    extent[[2, 3]] += grid_center[1]
    return extent


def show_pierre_auger(axis=None):
    pierre_auger = detector.generate_pierre_auger()
    grid_size = 100_000
    max_values = _np.max(pierre_auger.centers, axis=0)
    min_values = _np.max(pierre_auger.centers, axis=0)
    grid_center = (max_values + min_values) / 2
    extent = calculate_extent(grid_center, grid_size)
    add_scatter(pierre_auger, extent, axis=axis, alpha=0.5, s=1)


def show_subshower(inital_KE, energy_per_detector, array=None, axis=None, **kwargs):
    if array is None:
        array = detector.generate_pierre_auger()
    if axis is None:
        axis = _plt.gca()

    grid_center, grid_size = detector.center_and_size(array, energy_per_detector)
    grid_pa, grid_center, grid_size = detector.grid_positions(
        array, bins=1000, grid_size=grid_size, grid_center=grid_center
    )
    energy_grid = _np.copy(grid_pa).astype(float)
    for j, e in enumerate(energy_per_detector):
        energy_grid[grid_pa == (j + 1)] = e
    axis.set_title(
        f"initial KE {inital_KE:.2}, total deposit {energy_per_detector.sum():.4}"
    )
    extent = calculate_extent(grid_center, grid_size)
    path_collection = add_scatter(
        array,
        extent,
        energy_per_detector,
        alpha=0.5,
        s=20,
        cmap="cool",
        axis=axis,
        **kwargs,
    )
    if path_collection is not None:
        _plt.colorbar(path_collection)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


# Main workflow:
if __name__ == "__main__":
    relations = True

    # Create the figure (this holds the axis)
    fig = _go.Figure()

    if len(_sys.argv) < 2:
        print("Usage: python subshowers.py saved_showers [x_axis] [y_axis]")
    saved_showers = _sys.argv[1]
    if len(_sys.argv) > 2:
        x = _sys.argv[2]
        y = _sys.argv[3]
    else:
        x = "x"
        y = "z"

    from subshowers import Output

    output = Output.load(saved_showers)

    if relations:
        overview = Overview(output.shower_starts.reader, x_axis=x, y_axis=y)
        overview.relations_to_hist()
        add_shower_relations(fig, overview)
    else:
        add_shower_skeleton(fig, output.shower_starts.reader, x_axis=x, y_axis=y)

    add_selected_subshowers(fig, output.subshowers, output.shower_starts.reader)

    # Update layout (optional)

    fig.update_layout(
        title=f"File {saved_showers}",
        xaxis_title=x,
        yaxis_title=y,
        coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"),
    )

    # Save the figure
    input_base_name = _os.path.basename(saved_showers)[:-3]
    name = f"{input_base_name}_sample"
    if relations:
        name += "_relations"
    fig.write_html(f"{name}.html")
