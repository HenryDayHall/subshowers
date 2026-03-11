import os as _os
import plotly.graph_objects as _go
import numpy as _np
from plotly.express.colors import sample_colorscale


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
    chosen = _np.random.choice(
        len(subshowers), size=n_subshowers, replace=False
    )
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


# Main workflow:
if __name__ == "__main__":

    # Create the figure (this holds the axis)
    fig = _go.Figure()

    x = "x"
    y = "z"
    saved_showers = "/data/dust/user/dayhallh/eas/data/example_10e4_photon_hist1/subshowers_e1000.h5"
    from subshowers import Output

    output = Output.load(saved_showers)

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
    fig.write_html(f"{input_base_name}_sample.html")
