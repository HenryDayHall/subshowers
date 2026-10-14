import os

import numpy as np
import pytest
import yaml

from subshowers import detector


def test_sign_of_volume():
    square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    assert detector.sign_of_volume(*square) == 0
    positive_tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert detector.sign_of_volume(*positive_tetrahedron) == 1
    rotated_positive = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert detector.sign_of_volume(*rotated_positive) == 1
    negative_tetrahedron = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]])
    assert detector.sign_of_volume(*negative_tetrahedron) == -1


def test_intersects_triangle():
    point_1 = np.array([0, 0, 0])
    point_2 = np.array([0, 0, 10])
    triangle = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]])
    assert detector.intersects_triangle(point_1, point_2, triangle)
    triangle[:, 2] = -1
    assert not detector.intersects_triangle(point_1, point_2, triangle)
    point_1[2] = -10
    assert detector.intersects_triangle(point_1, point_2, triangle)
    point_1[2] = 0

    triangle[:, 2] = 1
    triangle[:, 0] += 10
    assert not detector.intersects_triangle(point_1, point_2, triangle)

    point_1 = np.array([-10, -10, -10])
    point_2 = np.array([10, 10, 10])
    triangle = np.array([[0, 1, 0], [-1, -1, 0], [1, -1, 0]])

    assert detector.intersects_triangle(point_1, point_2, triangle)
    triangle = np.array([[0, 0, 1], [0, -1, -1], [0, 1, -1]])
    assert detector.intersects_triangle(point_1, point_2, triangle)


def test_get_perpendiculars():
    vector_3d = np.array([0, 0, 1])
    perpendicular_1, perpendicular_2 = detector.get_perpendiculars(vector_3d)
    assert np.dot(perpendicular_1, perpendicular_2) < 1e-5
    assert np.dot(vector_3d, perpendicular_1) < 1e-5
    assert np.dot(vector_3d, perpendicular_2) < 1e-5
    vector_3d = np.array([0, 0, -1])
    perpendicular_1, perpendicular_2 = detector.get_perpendiculars(vector_3d)
    assert np.dot(perpendicular_1, perpendicular_2) < 1e-5
    assert np.dot(vector_3d, perpendicular_1) < 1e-5
    assert np.dot(vector_3d, perpendicular_2) < 1e-5


def test_horizontal_circle_hits():
    circles = detector.HorizontalCircles(
        np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 50.0]]), 10.0
    )
    starts = np.array(
        [[5.0, 0, 1000], [0, 0, 1000], [100, 8, 1000], [0, 0, -5], [0, 0, 1000]]
    )
    directions = np.array(
        [[0.0, 0, -1], [0.1, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, 1]]
    )
    hits = circles.hit_any(starts, directions)
    assert hits.tolist() == [
        [True, False],  # straight down, 5 m from the first centre
        [False, True],  # slanted, reaching the raised second circle 5 m off
        [False, True],
        [False, False],  # below both
        [False, False],  # going up
    ]


def test_grid_positions_of_triangles():
    """Both ways of drawing an array on a grid agree, for triangles too."""
    edges = np.array(
        [
            [[-30, -30, 0], [-10, -30, 0], [-20, -10, 0]],
            [[10, 10, 0], [35, 12, 0], [20, 30, 0]],
        ],
        dtype=float,
    )
    triangles = detector.Triangles(edges)
    bins, size = 100, 80.0
    perspective = np.array([0.0, 0.0, -1.0])
    x_increment = np.array([size / bins, 0, 0])
    y_increment = np.array([0, size / bins, 0])
    start_offset = -perspective - (x_increment + y_increment) * (bins / 2)
    brute_force = detector.brute_force_grid_positions(
        bins, 200, 1, perspective, start_offset, x_increment, y_increment, triangles
    )
    expanding = detector.expanding_grid_positions(
        bins, perspective, start_offset, x_increment, y_increment, triangles
    )
    assert (brute_force == 1).sum() > 50 and (brute_force == 2).sum() > 50
    assert np.array_equal(brute_force, expanding)


def test_pierre_auger_arrays():
    sizes = {None: 1689, "stop": 1636, "sd1500": 1640, "sd750": 71}
    for for_array, size in sizes.items():
        array = detector.generate_pierre_auger(for_array)
        assert len(array) == size
        # centred on the stations chosen
        assert np.allclose(array.centers[:, :2].mean(axis=0), 0, atol=1e-6)
    with pytest.raises(ValueError):
        detector.generate_pierre_auger("start")


def test_from_config():
    auger = {"detectors": {"generator": {"name": "generate_pierre_auger"}}}
    assert len(detector.from_config(auger)) == 1689
    assert not detector.is_unseeded(auger)
    auger["detectors"]["generator"]["for_array"] = "sd750"
    assert len(detector.from_config(auger)) == 71

    ground = {
        "detectors": {
            "generator": {"name": "generate_at_ground", "total": 7, "seed": 3}
        }
    }
    first, second = detector.from_config(ground), detector.from_config(ground)
    assert len(first) == 7
    assert np.array_equal(first.edges, second.edges)
    assert not detector.is_unseeded(ground)
    del ground["detectors"]["generator"]["seed"]
    assert detector.is_unseeded(ground)


def test_default_config_detector():
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    array = detector.from_config(config)
    assert len(array) == config["detectors"]["generator"]["total"]
    assert not detector.is_unseeded(config)


@pytest.mark.parametrize(
    "generator, message",
    [
        (None, "needs a detectors section"),
        ({"name": "generate_nothing"}, "name must be one of"),
        ({"name": "generate_at_ground"}, "total"),
        ({"name": "generate_at_ground", "total": 3, "colour": 1}, "colour"),
        ({"name": "generate_pierre_auger", "for_array": "sd3000"}, "for_array"),
    ],
)
def test_bad_detector_configs(generator, message):
    config = {} if generator is None else {"detectors": {"generator": generator}}
    with pytest.raises(detector.DetectorConfigError, match=message):
        detector.from_config(config)
