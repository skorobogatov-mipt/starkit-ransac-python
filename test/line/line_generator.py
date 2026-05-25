import pytest
from conftest import RNG
import numpy as np
from starkit_ransac.generators import generate_line3d
from starkit_ransac.surfaces import Line3D
from starkit_ransac.ransac_3d import RANSAC

class LineGenerator():
    # Numbers of variants of parameters
    N_POINTS = 5
    N_DIRECTIONS = 5

    # Generate random points and directions for testing
    points_list = (RNG.random((N_POINTS, 3)) * 20).tolist()
    directions_list = RNG.random((N_DIRECTIONS, 3))
    directions_list = (
        directions_list / np.linalg.norm(directions_list, axis=1)[:, np.newaxis]
    )
    directions_list = directions_list.tolist()

    SHAPE_NAME = 'line'

    @pytest.fixture(scope="class", params=points_list)
    def point(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=directions_list)
    def direction(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class")
    def perfect_model(self, point, direction):
        return Line3D(direction=direction, point=point)

    @pytest.fixture(scope="class", params=[0.0, 0.01, 0.02, 0.05, 0.1])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[5000, 2500, 1000, 500])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def data_points(self, perfect_model, noise_sigma, n_points):
        return generate_line3d(
            perfect_model, noise_sigma=noise_sigma, n_points=n_points
        )

    @pytest.fixture(scope="class")
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(Line3D, iter_num=1000, distance_threshold=0.1)
        return model

