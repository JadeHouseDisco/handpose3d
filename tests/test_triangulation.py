import pytest

np = pytest.importorskip("numpy")

from utils import DLT, triangulate_points


@pytest.fixture
def synthetic_triangulation_setup():
    """Provide synthetic projections and 2D observations for a known 3D point."""
    world_point = np.array([1.2, -0.5, 5.0])

    camera_centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    rotations = [np.eye(3) for _ in camera_centers]
    projections = []
    observations = []

    homogeneous_world = np.append(world_point, 1.0)

    for rotation, center in zip(rotations, camera_centers):
        translation = -rotation @ center
        projection = np.hstack([rotation, translation.reshape(3, 1)])
        projections.append(projection)

        projected = projection @ homogeneous_world
        observations.append(projected[:2] / projected[2])

    return projections, observations, world_point


def test_triangulate_with_three_cameras(synthetic_triangulation_setup):
    projections, observations, expected_point = synthetic_triangulation_setup

    reconstructed = triangulate_points(projections, observations)

    assert np.allclose(reconstructed, expected_point, atol=1e-6)


def test_triangulate_with_two_cameras_backward_compatible(synthetic_triangulation_setup):
    projections, observations, expected_point = synthetic_triangulation_setup

    reconstructed = triangulate_points(projections[:2], observations[:2])
    reconstructed_dlt = DLT(projections[0], projections[1], observations[0], observations[1])

    assert np.allclose(reconstructed, expected_point, atol=1e-6)
    assert np.allclose(reconstructed_dlt, expected_point, atol=1e-6)


def test_triangulate_with_missing_observation(synthetic_triangulation_setup):
    projections, observations, expected_point = synthetic_triangulation_setup
    invalid = (-1.0, -1.0)

    observations_with_gap = [observations[0], invalid, observations[2]]

    reconstructed = triangulate_points(projections, observations_with_gap, invalid_sentinel=invalid)

    assert np.allclose(reconstructed, expected_point, atol=1e-6)
