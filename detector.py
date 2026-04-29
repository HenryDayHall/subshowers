import numpy as np


def sign_of_volume(a, b, c, d):
    """
    Get the sign of the volumes made by the last dimension
    """
    cross_product = np.cross(b - a, c - a, axisa=-1)
    dot_product = np.sum((d - a) * cross_product, axis=-1)
    return np.sign(dot_product)


def intersects_triangle(point1, point2, triangle):
    sign_1 = sign_of_volume(
        point1, triangle[..., 0], triangle[..., 1], triangle[..., 2]
    )
    sign_2 = sign_of_volume(
        point2, triangle[..., 0], triangle[..., 1], triangle[..., 2]
    )

    requirement_1 = sign_1 != sign_2

    sign_3 = sign_of_volume(point1, point2, triangle[..., 0], triangle[..., 1])
    sign_4 = sign_of_volume(point1, point2, triangle[..., 1], triangle[..., 2])

    requirement_2 = sign_3 == sign_4

    sign_5 = sign_of_volume(point1, point2, triangle[..., 2], triangle[..., 0])

    requirement_3 = sign_5 == sign_4

    return requirement_1 & requirement_2 & requirement_3


class Triangles:
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
        self._centers = np.sum(edges, axis=-2) / 3
        # dimensions like (1, N, 3)
        self._centers_expanded = self._centers[None, :, :]

    def __len__(self):
        return self._length

    def hit(self, start_positions, directions):
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
        # dimensions like (M, N)
        distance_to_centers = np.sqrt(
            np.sum((self._centers_expanded - start_positions) ** 2, axis=-1)
        )
        # dimensions like (M, 1, 3)
        normed_directions = directions / np.sum(directions, axis=-1, keepdims=True)
        # dimensions like (M, N, 3)
        point_beyond = (
            start_positions + normed_directions * distance_to_centers[:, :, None] * 2
        )
        # dimensions like (M, N)
        intersects = intersects_triangle(
            start_positions, point_beyond, self._edges_expanded
        )
        return intersects


def generate_at_ground(
    total, distance_from_center=1000.0, mean_width=1.0, std_width=1.0
):
    angles = np.random.uniform(0, 2 * np.pi, size=(total))
    radial_distances = np.sqrt(
        np.random.uniform(0, distance_from_center**2, size=(total))
    )
    x_offset = radial_distances * np.cos(angles)
    y_offset = radial_distances * np.sin(angles)
    offset = np.stack((x_offset, y_offset), axis=-1)
    relative_edges = np.random.normal(0, 1, size=(total, 3, 2))
    unnormed_width_2 = np.maximum(
        np.maximum(
            np.sum((relative_edges[:, 0] - relative_edges[:, 1])**2, axis=-1),
            np.sum((relative_edges[:, 1] - relative_edges[:, 2])**2, axis=-1),
        ),
        np.sum((relative_edges[:, 2] - relative_edges[:, 0])**2, axis=-1),
    )
    unnormed_width = np.sqrt(unnormed_width_2)
    widths = np.random.normal(mean_width, std_width, size=(total, ))
    scale = widths / unnormed_width
    edges = scale[:, None, None] * relative_edges + offset[:, None, :]
    z_coord = np.zeros((total, 3, 1))
    edges = np.concatenate((edges, z_coord), axis=-1)
    return Triangles(edges)


def get_perpendiculars(vector_3d):
    normed = vector_3d / np.linalg.norm(vector_3d)
    if normed[2] == 1:
        one_perpendicular = np.array([1, 0, 0])
        two_perpendicular = np.array([0, 1, 0])
    else:
        one_perpendicular = np.cross(normed, np.array([0, 0, 1]))
        two_perpendicular = np.cross(one_perpendicular, normed)
    return one_perpendicular, two_perpendicular


def grid_positions(
    triangles, perspective=(0, 0, -1), grid_center=None, grid_size=None, bins=50
):
    perspective = np.array(perspective, dtype=float)
    # dimensions (N, 3, 3)
    edges = triangles.edges
    flat_edges = edges.reshape(-1, 3)
    along_perspective_direction = np.sum(flat_edges * perspective, axis=-1)

    if grid_center is None:
        max_values = np.max(flat_edges, axis=0)
        min_values = np.min(flat_edges, axis=0)
        grid_center = (max_values + min_values) / 2
    if grid_size is None:
        max_distance = np.max(np.sqrt(np.sum(flat_edges**2, axis=-1)))
        grid_size = max_distance * 2

    increment_size = grid_size / bins
    x_dir, y_dir = get_perpendiculars(perspective)
    x_increment = x_dir * increment_size
    y_increment = y_dir * increment_size

    if np.any(along_perspective_direction < 0):
        start_offset = np.min(along_perspective_direction) * 2 * perspective
    else:
        start_offset = -perspective
    start_offset += grid_center - (x_increment + y_increment) * (bins / 2)

    starting_points = np.tile(start_offset, (bins, bins, 1))
    x_shifts = x_increment[None, None, :] * np.arange(bins)[None, :, None]
    y_shifts = y_increment[None, None, :] * np.arange(bins)[:, None, None]

    starting_points += x_shifts + y_shifts
    flat_starting_points = starting_points.reshape(-1, 3)

    flat_perspective = np.tile(perspective, (bins * bins, 1))
    hit_triangles = triangles.hit(flat_starting_points, flat_perspective)
    hit_a_triangle = np.any(hit_triangles, axis=-1)

    return hit_a_triangle.reshape(bins, bins)


# TODO test
if __name__ == "__main__":
    triangles = generate_at_ground(20)
    grid = grid_positions(triangles)
    print(grid)
