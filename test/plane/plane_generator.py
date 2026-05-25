import pytest
from conftest import RNG
import numpy as np
from starkit_ransac.generators.plane import generate_plane
from starkit_ransac.surfaces.plane import Plane3D
from starkit_ransac.ransac_3d import RANSAC

class PlaneGenerator():
    approx_center_list = [
            [10, 54.2, 100], [0, 0, 0], [-4.2, 5.76, -1.228],
    ]
    coeffs_list = [
        [1, 1, 1, 0],
        [0, 2, 5, 10],
        [-345, 0, 1235, 23.1],
        [-0.054, 123, 0, 482.5748],
        [-0.8, 0.9, 3.57, 5.423],
    ]

    SHAPE_NAME = 'plane'

    @pytest.fixture(scope="class", params=coeffs_list)
    def coeffs(self, request):
        return np.array(request.param, float)

    @pytest.fixture(scope="class", params=approx_center_list)
    def approx_center(self, request):
        return np.array(request.param, float)

    @pytest.fixture(scope="class")
    def perfect_model(self, coeffs):
        return Plane3D(*coeffs)

    @pytest.fixture(scope="class", params=[0.0, 0.01, 0.02, 0.05, 0.1])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[5000, 2500, 1000, 500])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def data_points(self, perfect_model, noise_sigma, n_points, approx_center):
        return generate_plane(
            perfect_model,
            noise_sigma=noise_sigma,
            n_points=n_points,
            approx_center=approx_center,
            plane_size=1,
        )

    @pytest.fixture(scope="class")
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(Plane3D, iter_num=1000, distance_threshold=0.1)
        return model
