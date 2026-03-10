import pytest
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.line3d import Line3D
from starkit_ransac.generators.line3d import generate_line3d
from conftest import RNG

class TestLine3D:
    # Numbers of variants of parameters
    N_POINTS = 5
    N_DIRECTIONS = 5

    # Generate random points and directions for testing
    points_list = (RNG.random((N_POINTS, 3)) * 20).tolist()
    directions_list = RNG.random((N_DIRECTIONS, 3))
    directions_list = directions_list / np.linalg.norm(directions_list, axis=1)[:, np.newaxis]
    directions_list = directions_list.tolist()

    @pytest.fixture(scope='class', params=points_list)
    def point(self, request):
        return np.array(request.param)

    @pytest.fixture(scope='class', params=directions_list)
    def direction(self, request):
        return np.array(request.param)

    @pytest.fixture(scope='class')
    def perfect_model(self, point, direction):
        return Line3D(direction=direction, point=point)

    @pytest.fixture(
        scope='class',
        params=[0., 0.01, 0.02, 0.05, 0.1]
    )
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(
        scope='class',
        params=[5000, 2500, 1000, 500]
    )
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def data_points(self, perfect_model, noise_sigma, n_points):
        return generate_line3d(
            perfect_model,
            noise_sigma=noise_sigma,
            n_points=n_points
        )

    @pytest.fixture(scope='class')
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(
            Line3D,
            iter_num=1000,
            distance_threshold=0.1
        )
        return model

    @pytest.fixture()
    def acceptable_direction_rmse(self):
        return 0.1

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.2

    def test_direction_is_close(self, fit_model, perfect_model, acceptable_direction_rmse):
        actual = perfect_model.direction
        fit = fit_model.direction

        # Directions can be in opposite directions, so we check both
        pos_diff = np.linalg.norm(actual - fit)
        neg_diff = np.linalg.norm(actual + fit)
        min_diff = min(pos_diff, neg_diff)

        assert min_diff < acceptable_direction_rmse

    def test_point_is_close(self, fit_model, perfect_model, acceptable_point_rmse):
        # Project the perfect model's point onto the fit model's line
        # to get the closest point on the fit line
        a = fit_model.point
        v = fit_model.direction
        p = perfect_model.point

        # Calculate the projection of (p - a) onto v
        projection = np.dot(p - a, v) * v
        closest_point = a + projection

        # Calculate distance between perfect point and closest point on fit line
        distance = np.linalg.norm(p - closest_point)
        assert distance < acceptable_point_rmse

    def test_overall_close(
            self,
            perfect_model: Line3D,
            fit_model: Line3D,
            acceptable_point_rmse
        ):
        # Generate points on the perfect line
        perfect_points = generate_line3d(perfect_model, n_points=5000, noise_sigma=0)

        # Calculate distances from these points to the fit line
        distances = fit_model.calc_distances(perfect_points)
        rmse = np.sqrt(np.mean(np.sum(distances**2, axis=1)))

        assert rmse < acceptable_point_rmse

    def test_fit_model_method(self):
        # Test the fit_model method directly
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0]
        ])

        line = Line3D()
        line.fit_model(points)

        # Check if direction is along x-axis
        assert np.allclose(line.direction, [1, 0, 0])

        # Check if point is at origin
        assert np.allclose(line.point, [0, 0, 0])

    def test_calc_distances_method(self):
        # Test the calc_distances method
        line = Line3D(direction=[1, 0, 0], point=[0, 0, 0])

        # Points on the line should have zero distance
        points_on_line = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0]
        ])
        distances = line.calc_distances(points_on_line)
        assert np.allclose(distances, 0)

        # Points not on the line should have non-zero distance
        points_off_line = np.array([
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0]
        ])
        distances = line.calc_distances(points_off_line)
        assert np.allclose(np.linalg.norm(distances, axis=1), 1)
