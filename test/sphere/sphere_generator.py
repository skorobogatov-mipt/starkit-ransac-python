import pytest
from conftest import RNG
import numpy as np
from starkit_ransac.generators import generate_sphere
from starkit_ransac.surfaces import Sphere
from starkit_ransac.ransac_3d import RANSAC


class SphereGenerator():
    # Numbers of variants of parameters
    N_CENTERS = 5
    N_RADII = 5
    MAX_OFFSET = 20
    MAX_RADIUS = 5

    # Generate random centers and radii for testing
    centers_list = (RNG.random((N_CENTERS, 3)) * MAX_OFFSET).tolist()
    radii_list = (
        np.abs(RNG.random(N_RADII) * MAX_RADIUS) + 0.1  # ensure radius > 0
    ).tolist()

    SHAPE_NAME = 'sphere'

    @pytest.fixture(scope="class", params=centers_list)
    def center(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=radii_list)
    def radius(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def perfect_model(self, center, radius):
        return Sphere(center, radius)

    @pytest.fixture(scope="class", params=[0.0, 0.01, 0.02, 0.05, 0.1])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[5000, 2500, 1000, 500])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def data_points(self, perfect_model, noise_sigma, n_points):
        return generate_sphere(
            perfect_model, noise_sigma=noise_sigma, n_points=n_points
        )

    @pytest.fixture(scope="class")
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(Sphere, iter_num=5000, distance_threshold=0.1)
        return model
