import numpy as np
import pytest
from conftest import BENCHMARK_THRESH, N_ITER_BENCHMARK, SEED, RNG
from starkit_ransac.generators.circle import generate_circle
from starkit_ransac.surfaces.circle import Circle3D
from starkit_ransac.ransac_3d import RANSAC

class Circle3DGenerator():
    MAX_OFFSET = 20
    center_coordinates = (RNG.random((3, 3)) * MAX_OFFSET).tolist()

    normals = RNG.random((3, 3)).tolist()

    MAX_RADIUS = 5
    radii = (RNG.random(3) * MAX_RADIUS).tolist()

    SHAPE_NAME = 'circle3d'

    @pytest.fixture(scope="class", params=center_coordinates)
    def center(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=radii)
    def radius(self, request):
        return request.param

    @pytest.fixture(scope="class", params=normals)
    def normal(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def perfect_model(self, center, radius, normal):
        return Circle3D(center=center, radius=radius, normal=normal)

    @pytest.fixture(scope="class", params=[0, 0.05, 0.1, 0.5])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1000, 500, 250])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def data_points(self, perfect_model, noise_sigma, n_points):
        data = generate_circle(
            perfect_model, noise_sigma=noise_sigma, n_points=n_points
        )
        return data

    @pytest.fixture(scope="class")
    def fitted_circle(self, circle_data):
        ransac = RANSAC()
        ransac.add_points(circle_data)

        model = ransac.fit(Circle3D, 500, 0.1)
        return model

