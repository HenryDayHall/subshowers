import inspect
import numpy as np
import os
from abc import ABC


def sign_of_volume(a, b, c, d):
    """
    Get the sign of the volumes made by the last dimension
    """
    cross_product = np.cross(b - a, c - a, axisa=-1)
    dot_product = np.sum((d - a) * cross_product, axis=-1)
    return np.sign(dot_product)


def intersects_triangle(point1, point2, triangle):
    """
    Check if the cord connecting 2 points intersects a triangle.
    The initial unspecified dimensions of each array match.

    Parameters
    ----------
    point1 : array (..., 3)
    point2 : array (..., 3)
    triangle : array (..., 3, 3)

    Returns
    -------
    intersects : array (...)
    """
    sign_1 = sign_of_volume(
        point1, triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]
    )
    sign_2 = sign_of_volume(
        point2, triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]
    )

    requirement_1 = sign_1 != sign_2

    sign_3 = sign_of_volume(point1, point2, triangle[..., 0, :], triangle[..., 1, :])
    sign_4 = sign_of_volume(point1, point2, triangle[..., 1, :], triangle[..., 2, :])

    requirement_2 = sign_3 == sign_4

    sign_5 = sign_of_volume(point1, point2, triangle[..., 2, :], triangle[..., 0, :])
    requirement_3 = sign_5 == sign_4

    return requirement_1 & requirement_2 & requirement_3


def intersects_horizontal_circle(start_position, normed_direction, centers, radius2):
    """
    Check if a half line, starting at a specified height, intersects a horizontal circle.
    The initial unspecified dimensions of each array match.

    Parameters
    ----------
    start_position : array (..., 3)
    normed_direction : array (..., 3)
    centers : array (..., 3)
    radius2 : array (...)

    Returns
    -------
    intersects : array (...)
    """
    distance_to_plane = (start_position[..., 2] - centers[..., 2]) / -normed_direction[
        ..., 2
    ]
    # offset in x and y, from each center, of where the line meets its plane
    x_offset = (
        start_position[..., 0] + normed_direction[..., 0] * distance_to_plane
    ) - centers[..., 0]
    y_offset = (
        start_position[..., 1] + normed_direction[..., 1] * distance_to_plane
    ) - centers[..., 1]
    in_range = x_offset * x_offset + y_offset * y_offset <= radius2
    into_plane = distance_to_plane > 0
    return in_range & into_plane


class SensitiveRegions(ABC):
    def __len__(self):
        raise NotImplementedError

    def hit_any(self, start_positions, directions):
        """
        Determine which of the triangles are hit by which vectors

        Parameters
        ----------
        start_positions : array (M, 3)
        directions : array (M, 3)
        """
        assert start_positions.shape == directions.shape
        assert start_positions.shape[-1] == 3
        assert len(start_positions.shape) == 2
        # add an extra dimension to the start_positions and directions
        # dimensions like (M, 1, 3)
        start_positions = start_positions[..., None, :]
        # dimensions like (M, 1, 3)
        directions = directions[..., None, :]
        # dimensions like (M, 1, 3)
        normed_directions = directions / np.linalg.norm(
            directions, axis=-1, keepdims=True
        )
        # dimensions like (M, N)
        intersects = self._all_check_intersections(start_positions, normed_directions)
        return intersects

    def hit_idxs(self, idxs, start_positions, directions):
        """
        Determine which of the triangles are hit by which vectors

        Parameters
        ----------
        idxs : array (M)
        start_positions : array (M, 3)
        directions : array (M, 3)
        """
        assert start_positions.shape == directions.shape
        assert start_positions.shape[-1] == 3
        assert len(start_positions.shape) == 2
        assert len(idxs) == len(start_positions)
        # add an extra dimension to the start_positions and directions
        # dimensions like (M, 3)
        normed_directions = directions / np.linalg.norm(
            directions, axis=-1, keepdims=True
        )
        # dimensions like (M, N)
        intersects = self._these_check_intersections(
            idxs, start_positions, normed_directions
        )
        return intersects

    def _these_check_intersections(self, start_positions, normed_directions):
        raise NotImplementedError

    def _all_check_intersections(self, start_positions, normed_directions):
        raise NotImplementedError


class Triangles(SensitiveRegions):
    def __init__(self, edges):
        """
        A set of triangular regions where the upper surface is sensitive

        Parameters
        ----------
        edges : array (N, 3, 3)
        """
        self._length = len(edges)
        assert edges.shape == (self._length, 3, 3)
        # dimensions like (N, 3, 3)
        self.edges = edges
        # dimensions like (1, N, 3, 3)
        self._edges_expanded = edges[None, :, :, :]
        # dimensions like (N, 3)
        self.centers = np.sum(edges, axis=-2) / 3
        # dimensions like (1, N, 3)
        self._centers_expanded = self.centers[None, :, :]

    def __len__(self):
        return self._length

    def _all_check_intersections(self, start_positions, normed_directions):
        # dimensions like (M, N)
        distance_to_centers = np.sqrt(
            np.sum((self._centers_expanded - start_positions) ** 2, axis=-1)
        )
        # dimensions like (M, N, 3)
        point_beyond = (
            start_positions + normed_directions * distance_to_centers[:, :, None] * 2
        )
        intersects = intersects_triangle(
            start_positions, point_beyond, self._edges_expanded
        )
        return intersects

    def _these_check_intersections(self, idxs, start_positions, normed_directions):
        # one triangle for each start position; dimensions like (M,)
        distance_to_centers = np.sqrt(
            np.sum((self.centers[idxs] - start_positions) ** 2, axis=-1)
        )
        # dimensions like (M, 3)
        point_beyond = (
            start_positions + normed_directions * distance_to_centers[:, None] * 2
        )
        intersects = intersects_triangle(
            start_positions, point_beyond, self.edges[idxs]
        )
        return intersects


class HorizontalCircles(SensitiveRegions):
    def __init__(self, centers, radius):
        """
        A set of circular regions where the upper surface is sensitive

        Parameters
        ----------
        edges : array (N, 3, 3)
        """
        self._length = len(centers)
        self.centers = centers
        self._centers_expanded = centers[None, :, :]
        self.radius = radius
        if not hasattr(radius, "__len__"):
            self._projected_radii = np.full(self._length, radius)
        elif len(radius) == 1:
            self._projected_radii = np.full(self._length, radius[0])
        else:
            assert len(radius) == self._length
            self._projected_radii = radius
        self.radii2 = self._projected_radii**2
        self._radii2_expanded = self.radii2[None, :]

    def __len__(self):
        return self._length

    def _all_check_intersections(self, start_positions, normed_directions):
        intersects = intersects_horizontal_circle(
            start_positions,
            normed_directions,
            self._centers_expanded,
            self._radii2_expanded,
        )
        return intersects

    def _these_check_intersections(self, idxs, start_positions, normed_directions):
        intersects = intersects_horizontal_circle(
            start_positions,
            normed_directions,
            self.centers[idxs],
            self.radii2[idxs],
        )
        return intersects


def generate_at_ground(
    total, distance_from_center=1000.0, mean_width=1.0, std_width=1.0, seed=None
):
    """
    `total` triangles at z = 0, placed at random within distance_from_center
    of the origin. With a seed, the same array is made every time; without
    one, numpy's global random state (np.random.seed) decides.
    """
    rng = np.random if seed is None else np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=(total))
    radial_distances = np.sqrt(rng.uniform(0, distance_from_center**2, size=(total)))
    x_offset = radial_distances * np.cos(angles)
    y_offset = radial_distances * np.sin(angles)
    offset = np.stack((x_offset, y_offset), axis=-1)
    relative_edges = rng.normal(0, 1, size=(total, 3, 2))
    unnormed_width_2 = np.maximum(
        np.maximum(
            np.sum((relative_edges[:, 0] - relative_edges[:, 1]) ** 2, axis=-1),
            np.sum((relative_edges[:, 1] - relative_edges[:, 2]) ** 2, axis=-1),
        ),
        np.sum((relative_edges[:, 2] - relative_edges[:, 0]) ** 2, axis=-1),
    )
    unnormed_width = np.sqrt(unnormed_width_2)
    widths = rng.normal(mean_width, std_width, size=(total,))
    scale = widths / unnormed_width
    edges = scale[:, None, None] * relative_edges + offset[:, None, :]
    z_coord = np.zeros((total, 3, 1))
    edges = np.concatenate((edges, z_coord), axis=-1)
    return Triangles(edges)


DEFAULT_COLUMNS = ["id", "northing", "easting", "altitude"]

# Columns of sdMap.csv after DEFAULT_COLUMNS. start holds each station's start
# time; stop is 1 for stations still running, and otherwise the time they
# stopped; sd1500 and sd750 are 1 for stations in those grids.
PIERRE_AUGER_COLUMNS = ["start", "stop", "sd1500", "sd750"]
# for_array options: the columns where 1 picks out a set of stations
PIERRE_AUGER_ARRAYS = ["stop", "sd1500", "sd750"]


def generate_pierre_auger(for_array=None):
    """
    The surface detector stations of the Pierre Auger Observatory, as circles
    of radius 10 m, centred on the mean position of the stations chosen.

    for_array picks the stations: "stop" for those still running, "sd1500"
    or "sd750" for the stations of those grids, or None for every station
    listed. That includes stations since replaced, some at the same place as
    their replacement, where a particle is counted twice.
    """
    if for_array is not None and for_array not in PIERRE_AUGER_ARRAYS:
        raise ValueError(
            f"for_array must be one of {PIERRE_AUGER_ARRAYS} or None, not {for_array!r}"
        )
    cols = DEFAULT_COLUMNS + PIERRE_AUGER_COLUMNS
    script_location = os.path.dirname(os.path.realpath(__file__))
    csv_location = os.path.join(script_location, "sdMap.csv")
    array = generate_from_csv(csv_location, cols, for_array)
    return array


def generate_from_csv(file_name, columns=None, filter_column=None, match_value=1):
    """
    Horizontal circles of radius 10 m at the stations listed in a csv file
    with columns id, northing, easting and altitude (and any others named in
    columns), centred on the mean easting and northing. With filter_column,
    only the stations where that column equals match_value are kept.
    """
    if columns is None:
        columns = DEFAULT_COLUMNS
    with open(file_name, "r") as f:
        csv_values = np.genfromtxt(f, delimiter=",", dtype=str, skip_header=1).T
    columns_in_file = csv_values.shape[0]
    assert columns_in_file == len(
        columns
    ), f"named column length {len(columns)} != number columns in file {columns_in_file}"
    if filter_column is not None:
        # csv_values has one row per column, and one column per station
        filter_on = csv_values[columns.index(filter_column)].astype(float)
        csv_values = csv_values[:, filter_on == match_value]
        if csv_values.shape[1] == 0:
            raise ValueError(
                f"no stations in {file_name} have {filter_column} = {match_value}"
            )
    centers = np.zeros((csv_values.shape[1], 3))
    xs = csv_values[columns.index("easting")].astype(float)
    xs -= np.mean(xs)
    centers[:, 0] = xs
    ys = csv_values[columns.index("northing")].astype(float)
    ys -= np.mean(ys)
    centers[:, 1] = ys
    zs = csv_values[columns.index("altitude")].astype(float)
    centers[:, 2] = zs
    return HorizontalCircles(centers, 10)


class DetectorConfigError(ValueError):
    """A config's detectors section doesn't describe an array that can be made."""


def generators():
    """The functions a config can name to build an array: {name: function}."""
    return {
        name: value
        for name, value in globals().items()
        if name.startswith("generate_") and callable(value)
    }


def generator_settings(config):
    """(name, arguments) from a config's detectors: generator section."""
    section = config.get("detectors") if isinstance(config, dict) else None
    settings = section.get("generator") if isinstance(section, dict) else None
    if not isinstance(settings, dict):
        raise DetectorConfigError(
            "the config needs a detectors section with a generator, e.g."
            " detectors: {generator: {name: generate_pierre_auger}}"
        )
    arguments = dict(settings)
    name = arguments.pop("name", None)
    options = generators()
    if not isinstance(name, str) or name not in options:
        raise DetectorConfigError(
            f"detectors: generator: name must be one of {sorted(options)},"
            f" not {name!r}"
        )
    try:
        inspect.signature(options[name]).bind(**arguments)
    except TypeError as error:
        raise DetectorConfigError(f"detectors: generator: {name}: {error}") from None
    return name, arguments


def from_config(config):
    """
    Build the array a config's detectors section describes. The generator's
    name is one of the generate_ functions here, and its other keys are that
    function's arguments:

        detectors:
          generator:
            name: generate_pierre_auger
            for_array: stop
    """
    name, arguments = generator_settings(config)
    try:
        return generators()[name](**arguments)
    except (TypeError, ValueError) as error:
        raise DetectorConfigError(f"detectors: generator: {name}: {error}") from None


def is_unseeded(config):
    """
    True if the config's detector array is random and has no seed, so every
    array made from the config is different.
    """
    name, arguments = generator_settings(config)
    parameters = inspect.signature(generators()[name]).parameters
    return "seed" in parameters and arguments.get("seed") is None


def spacing(array):
    """
    The usual distance (m) between neighbouring detectors of an array: the
    median, over the detectors, of the distance across the ground (in x and
    y) from each one's centre to the nearest other one's. About 1.5 km for
    the Pierre Auger surface detector, and 750 m for its sd750 grid.

    Detectors in the same place, like the Pierre Auger stations replaced
    where they stood, are counted once, and the array needs detectors in at
    least two places to have a spacing.
    """
    centres = np.unique(np.asarray(array.centers, dtype=float)[:, :2], axis=0)
    n_places = len(centres)
    if n_places < 2:
        raise ValueError(
            f"the array has detectors in {n_places} place{'' if n_places == 1 else 's'},"
            " so they have no spacing"
        )
    x, y = centres[:, 0], centres[:, 1]
    nearest2 = np.empty(n_places)
    # from some detectors at a time, so at most a few million distances are held
    step = max(1, 2_000_000 // n_places)
    for start in range(0, n_places, step):
        stop = min(start + step, n_places)
        distance2 = (x[start:stop, None] - x[None, :]) ** 2
        distance2 += (y[start:stop, None] - y[None, :]) ** 2
        # not from a detector to itself
        distance2[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest2[start:stop] = distance2.min(axis=1)
    return float(np.median(np.sqrt(nearest2)))


def max_shift(array):
    """
    The furthest (m) the sampler moves a subshower across the ground when it
    shifts subshowers to augment the data (see sampling.SubshowerSampler):
    the spacing of the array's detectors.

    A subshower is much the same wherever it falls, so shifting it changes
    only which detectors it reaches. No point of a square grid is further
    than 0.71 of a spacing from a detector, nor of a triangular grid, like
    the Pierre Auger surface detector, further than 0.58, so shifts of up to
    a spacing can put any part of a subshower over a detector: the subshower
    can land anywhere relative to the grid. Yet it stays near where it was
    simulated, in the same part of the array, and about as far from the core
    of its shower and from the edges of the array.
    """
    return spacing(array)


def edges_from_coords(northing, easting, altitude, single_detector, center=None):
    if center is None:
        center = np.mean([northing, easting], axis=0)
    xs = northing - center[0]
    ys = easting - center[1]
    zs = altitude
    n_detectors = len(xs)
    edges = np.tile(single_detector, (n_detectors, 1, 1, 1))
    edges[:, :, :, 0] += xs[:, None, None]
    edges[:, :, :, 1] += ys[:, None, None]
    edges[:, :, :, 2] += zs[:, None, None]
    edges = edges.reshape(-1, 3, 3)
    return edges


def get_perpendiculars(vector_3d):
    normed = vector_3d / np.linalg.norm(vector_3d)
    if np.abs(normed[2]) == 1:
        one_perpendicular = np.array([1, 0, 0])
        two_perpendicular = np.array([0, 1, 0])
    else:
        one_perpendicular = np.cross(normed, np.array([0, 0, 1]))
        two_perpendicular = np.cross(one_perpendicular, normed)
    return one_perpendicular, two_perpendicular


def grid_positions(
    sensitive_regions, perspective=(0, 0, -1), grid_center=None, grid_size=None, bins=50
):
    perspective = np.array(perspective, dtype=float)
    # dimensions (N, 3, 3)
    centers = np.copy(sensitive_regions.centers)
    along_perspective_direction = np.sum(centers * perspective, axis=-1)

    if grid_center is None:
        max_values = np.max(centers, axis=0)
        min_values = np.min(centers, axis=0)
        grid_center = (max_values + min_values) / 2
    if grid_size is None:
        max_distance = np.max(np.sqrt(np.sum(centers**2, axis=-1)))
        grid_size = max_distance * 2 * 1.05

    increment_size = grid_size / bins
    x_dir, y_dir = get_perpendiculars(perspective)
    x_increment = x_dir * increment_size
    y_increment = y_dir * increment_size

    if np.any(along_perspective_direction < 0):
        start_offset = np.min(along_perspective_direction) * 2 * perspective
    else:
        start_offset = -perspective
    start_offset += grid_center - (x_increment + y_increment) * (bins / 2)

    max_chunk_size = 200
    total_chunks_per_dim = np.ceil(bins / max_chunk_size)
    if total_chunks_per_dim < 3:
        output_grid = brute_force_grid_positions(
            bins,
            max_chunk_size,
            total_chunks_per_dim,
            perspective,
            start_offset,
            x_increment,
            y_increment,
            sensitive_regions,
        )
    else:
        output_grid = expanding_grid_positions(
            bins,
            perspective,
            start_offset,
            x_increment,
            y_increment,
            sensitive_regions,
        )

    return output_grid, grid_center, grid_size


def brute_force_grid_positions(
    bins,
    max_chunk_size,
    total_chunks_per_dim,
    perspective,
    start_offset,
    x_increment,
    y_increment,
    sensitive_regions,
):

    output_grid = np.zeros((bins, bins))
    flat_perspective = np.tile(perspective, (max_chunk_size * max_chunk_size, 1))
    total_chunks = total_chunks_per_dim**2

    for x_chunk, x_start in enumerate(range(0, bins, max_chunk_size)):
        for y_chunk, y_start in enumerate(range(0, bins, max_chunk_size)):
            if total_chunks > 1:
                percent_reached = (
                    x_chunk * total_chunks_per_dim + y_chunk
                ) / total_chunks
                print(f"{percent_reached:.0%}", end="\r", flush=True)
            chunk_size_x = min(max_chunk_size, bins - x_start)
            chunk_size_y = min(max_chunk_size, bins - y_start)
            x_end = x_start + chunk_size_x
            y_end = y_start + chunk_size_y

            starting_points = np.tile(start_offset, (chunk_size_y, chunk_size_x, 1))
            x_shifts = (
                x_increment[None, None, :] * np.arange(x_start, x_end)[None, :, None]
            )
            y_shifts = (
                y_increment[None, None, :] * np.arange(y_start, y_end)[:, None, None]
            )

            starting_points += x_shifts + y_shifts
            flat_starting_points = starting_points.reshape(-1, 3)
            flat_perspective_here = flat_perspective[: len(flat_starting_points)]

            hit_regions = sensitive_regions.hit_any(
                flat_starting_points, flat_perspective_here
            ).astype(int)

            hit_regions *= np.arange(1, len(sensitive_regions) + 1)

            hit_which_region = np.max(hit_regions, axis=-1).reshape(
                chunk_size_y, chunk_size_x
            )
            output_grid[y_start:y_end, x_start:x_end] = hit_which_region

    return output_grid


def closest_bin_to_centers(sensitive_regions, bin_xs, bin_ys):
    x_distances = np.abs(sensitive_regions.centers[:, [0]] - bin_xs[None, :])
    y_distances = np.abs(sensitive_regions.centers[:, [1]] - bin_ys[None, :])

    closest_x = np.argmin(x_distances, axis=1)
    closest_y = np.argmin(y_distances, axis=1)

    return closest_x, closest_y


def square_idxs(square_size):
    if square_size == 1:
        return np.array([0]), np.array([0])
    steps_out = square_size // 2
    side_length = steps_out * 2 + 1
    top_bottom_xs = np.arange(-steps_out, steps_out + 1)
    top_ys = np.full(side_length, steps_out)
    bottom_ys = np.full(side_length, -steps_out)

    left_right_ys = np.arange(-steps_out + 1, steps_out)
    left_xs = np.full(side_length - 2, -steps_out)
    right_xs = np.full(side_length - 2, steps_out)

    xs = np.concatenate([top_bottom_xs, left_xs, top_bottom_xs, right_xs])
    ys = np.concatenate([top_ys, left_right_ys, bottom_ys, left_right_ys])
    return xs, ys


def expanding_grid_positions(
    bins,
    perspective,
    start_offset,
    x_increment,
    y_increment,
    sensitive_regions,
):
    bin_xs = x_increment[0] * np.arange(bins) + start_offset[0]
    bin_ys = y_increment[1] * np.arange(bins) + start_offset[1]

    output_grid = np.zeros((bins, bins))
    keep_going = np.arange(len(sensitive_regions), dtype=int)
    closest_x, closest_y = closest_bin_to_centers(sensitive_regions, bin_xs, bin_ys)

    square_size = 1
    while keep_going.shape[0]:
        print(f"square size: {square_size}", end="\r", flush=True)
        relative_xs, relative_ys = square_idxs(square_size)
        n_to_check = len(relative_xs)
        x_bins = (closest_x[keep_going, None] + relative_xs[None, :]).flatten()
        y_bins = (closest_y[keep_going, None] + relative_ys[None, :]).flatten()
        mask = (x_bins >= 0) * (x_bins < bins) * (y_bins >= 0) * (y_bins < bins)
        x_bins = x_bins[mask]
        y_bins = y_bins[mask]
        x_locations = bin_xs[x_bins]
        y_locations = bin_ys[y_bins]
        total_locations = len(x_locations)
        start_positions = np.stack(
            [x_locations, y_locations, np.full(total_locations, start_offset[2])],
            axis=-1,
        )
        perspective_here = np.tile(perspective, (total_locations, 1))
        idxs = np.repeat(keep_going, n_to_check)[mask]
        hits = sensitive_regions.hit_idxs(idxs, start_positions, perspective_here)
        hit_idxs = idxs[hits]
        x_hit = x_bins[hits]
        y_hit = y_bins[hits]
        output_grid[y_hit, x_hit] = (hit_idxs+1)
        keep_going = np.unique(hit_idxs)
        square_size += 1

    return output_grid


def center_and_size(sensitive_regions, energy_per_region):
    active = energy_per_region > 0
    if not active.any():
        return None, None
    locations = sensitive_regions.centers[active]
    min_x, min_y, _ = np.min(locations, axis=0)
    max_x, max_y, _ = np.max(locations, axis=0)

    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, 0])
    size = np.max([max_x - min_x, max_y - min_y]) * 1.1
    return center, size


if __name__ == "__main__":
    thing = 2
    if thing == 1:
        triangles = generate_at_ground(20)
        grid = grid_positions(triangles)
        print(grid)
    elif thing == 2:
        pierre_auger = generate_pierre_auger()
        grid = grid_positions(
            pierre_auger, bins=3000, grid_size=1000, grid_center=pierre_auger.centers[0]
        )
        print(grid.sum())
