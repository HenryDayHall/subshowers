from subshowers import detector
import numpy as np


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


if __name__ == "__main__":
    test_sign_of_volume()
    test_intersects_triangle()
