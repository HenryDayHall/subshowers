"""
Find the location and energy of particles that reach a detection plane
"""
import numpy as np


def point_on_plane(start_position, normed_direction, height):
    """
    Find the location of intersection of a half line with a plane.

    Parameters
    ----------
    start_position : array (..., 3)
        Last dimension is x, y, z
    normed_direction : array (..., 3)
        Normalised to unit length
        Last dimension is dx, dy, dz
    height : float
        Height of the plane, z = height

    Returns
    -------
    intersect_mask : array (...):
        True if the half line intersects the plane
    position : array (..., 2)
        Location of intersection in x, y

    """
    distance_to_plane = (start_position[..., 2] - height) / -normed_direction[
        ..., 2
    ]
    position = (
        start_position[..., [0, 1]]
        + normed_direction[..., [0, 1]] * distance_to_plane[..., None]
    )
    intersect_mask = np.sign(distance_to_plane) > 0
    return intersect_mask, position


class PlaneDistributions:
    def __init__(self, height):
        self.height = height
        self.subshower_root_idxs = []
        self.subshower_root_kinematics = []
        self.distributions = []

    def add_subshowers_idxs(self, subshowers):
        self.subshower_idxs += list(subshower_idxs)
        start_positions = np.stack(
            [reader.data[c][subshower_idxs] for c in ["x", "y", "z"]], axis=-1
        )
        directions = np.stack(
            [reader.data[c][subshower_idxs] for c in ["px", "py", "pz"]], axis=-1
        )

