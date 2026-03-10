import numpy as np
import pytest
from starkit_ransac.generators.circle import generate_circle
from starkit_ransac.surfaces.circle import Circle
from starkit_ransac.ransac_3d import RANSAC
from conftest import SEED,RNG


class TestCircle3D:
    MAX_OFFSET = 20
    center_coordinates = (
        RNG.random((5, 3)) * MAX_OFFSET
    ).tolist()

    normals = RNG.random((5, 3)).tolist()

    MAX_RADIUS = 5
    radii = (
        RNG.random(5) * MAX_RADIUS
    ).tolist()

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
    def perfect_circle(self, center, radius, normal):
        return Circle(
            center=center,
            radius=radius,
            normal=normal
        )

    @pytest.fixture(scope="class", params=[0, 0.05, 0.1, 0.5])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1000, 500, 250])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def circle_data(self, perfect_circle, noise_sigma, n_points):
        data = generate_circle(
            perfect_circle,
            noise_sigma=noise_sigma,
            n_points=n_points
        )
        return data

    @pytest.fixture(scope="class")
    def fitted_circle(self, circle_data):
        ransac = RANSAC()
        ransac.add_points(circle_data)

        model = ransac.fit(
            Circle,
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

    @pytest.fixture(scope="class")
    def acceptable_normal_error(self):
        return 0.1

    def test_radii_are_close(
        self,
        fitted_circle,
        perfect_circle,
        acceptable_radius_error
    ):
        fit_radius = fitted_circle.radius
        actual_radius = perfect_circle.radius
        relative_radius_error = abs(fit_radius - actual_radius) / actual_radius
        assert relative_radius_error < acceptable_radius_error

    def test_normals_are_close(
        self,
        fitted_circle,
        perfect_circle,
        acceptable_normal_error
    ):
        fit_normal = fitted_circle.normal
        actual_normal = perfect_circle.normal
        dist1 = np.linalg.norm(fit_normal - actual_normal)
        dist2 = np.linalg.norm(fit_normal + actual_normal)
        assert min(dist1, dist2) < acceptable_normal_error

    def test_centers_are_close(
        self,
        fitted_circle,
        perfect_circle,
        acceptable_center_error
    ):
        fit_center = fitted_circle.center
        actual_center = perfect_circle.center
        dist = np.linalg.norm(fit_center - actual_center)
        assert dist < acceptable_center_error
