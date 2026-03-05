import numpy as np
import pytest
from starkit_ransac.generators.circle2d import generate_circle2D
from starkit_ransac.surfaces.circle2d import Circle2D
from starkit_ransac.ransac_3d import RANSAC3D
from conftest import SEED


class TestCircle2D:
    MAX_OFFSET = 20
    center_coordinates = (
        np.random.default_rng(SEED).random(5) * MAX_OFFSET
    ).tolist()
    MAX_RADIUS = 5
    radii = (
        np.random.default_rng(SEED).random(5) * MAX_RADIUS
    ).tolist()

    @pytest.fixture(scope="class", params=center_coordinates)
    def x(self, request):
        return request.param

    @pytest.fixture(scope="class", params=center_coordinates)
    def y(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def center(self, x, y):
        return np.array((x, y))

    @pytest.fixture(scope="class", params=radii)
    def radius(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def perfect_circle(self, center, radius):
        return Circle2D(
            center=center,
            radius=radius
        )

    @pytest.fixture(scope="class", params=[0, 0.05, 0.1, 0.5])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1000, 500, 250, 100, 50])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def circle_data(self, perfect_circle, noise_sigma, n_points):
        data = generate_circle2D(
            perfect_circle,
            noise_sigma=noise_sigma,
            n_points=n_points
        )
        return data

    @pytest.fixture(scope="class")
    def fitted_circle(self, circle_data):
        ransac = RANSAC3D()
        ransac.add_points(circle_data)

        model = ransac.fit(
            Circle2D,
            500,
            0.1
        )
        return model

    @pytest.fixture(scope="class")
    def acceptable_radius_error(self):
        return 0.05

    @pytest.fixture(scope="class")
    def acceptable_center_error(self):
        return 0.1

    def test_radii_are_close(
        self,
        fitted_circle,
        perfect_circle,
        acceptable_radius_error
    ):
        fit_radius = fitted_circle.model['radius']
        actual_radius = perfect_circle.model['radius']
        relative_radius_error = abs(fit_radius - actual_radius) / actual_radius
        assert relative_radius_error < acceptable_radius_error

    def test_centers_are_close(
        self,
        fitted_circle,
        perfect_circle,
        acceptable_center_error
    ):
        fit_center = fitted_circle.model['center']
        actual_center = perfect_circle.model['center']
        dist = np.linalg.norm(fit_center - actual_center)
        assert dist < acceptable_center_error
