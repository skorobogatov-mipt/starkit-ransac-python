import pytest
import numpy as np
from numpy.typing import ArrayLike, NDArray
from starkit_ransac.surfaces.sphere import Sphere
from starkit_ransac.ransac_3d import RANSAC
from conftest import RNG
from starkit_ransac.generators.sphere import generate_sphere
from pytest_benchmark.plugin import benchmark
import pyransac3d

class TestSphere:
    # numbers of variants of a parameter
    N_CENTERS = 5
    N_RADII = 5
    MAX_OFFSET = 20

    centers_list = (
        RNG.random((N_CENTERS, 3)) * MAX_OFFSET
    ).tolist()

    MAX_RADIUS = 5
    radii_list = (
        np.abs(RNG.random(N_RADII) * MAX_RADIUS) + 0.1  # ensure radius > 0
    ).tolist()

    @pytest.fixture(scope='class', params=centers_list)
    def center(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=radii_list)
    def radius(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def perfect_model(
            self,
            center,
            radius
        ):
        return Sphere(center, radius)

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
        return generate_sphere(
                perfect_model, 
                noise_sigma=noise_sigma, 
                n_points=n_points
        )

    @pytest.fixture(scope='class')
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(
            Sphere,
            5000,
            0.1
        )
        return model

    @pytest.fixture()
    def acceptable_radius_relative_error(self):
        return 0.1

    @pytest.fixture()
    def acceptable_center_distance(self):
        return 0.25

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.5

    def test_radius_is_close(self, fit_model, perfect_model, acceptable_radius_relative_error):
        actual = perfect_model.radius
        fit = fit_model.radius
        diff = np.abs((actual - fit) / actual)
        assert diff < acceptable_radius_relative_error

    def test_center_is_close(
            self,
            fit_model,
            perfect_model,
            acceptable_center_distance
        ):
        actual = perfect_model.center
        fit = fit_model.center
        diff = np.linalg.norm(actual - fit)
        assert diff < acceptable_center_distance

    def test_overall_close(
            self,
            perfect_model: Sphere,
            fit_model: Sphere,
            acceptable_point_rmse
        ):
        # Generate points on the perfect sphere
        n_points = 5000
        points = generate_sphere(
                perfect_model,
                n_points=n_points,
                noise_sigma=0
        )

        # Calculate distances to the fitted sphere
        distances = fit_model.calc_distances(points)
        rmse = np.sqrt(np.mean(distances**2))
        assert rmse < acceptable_point_rmse

    def test_fit_model_with_minimal_points(self):
        # Test that the model can be fit with the minimal number of points
        center = np.array([1.0, 2.0, 3.0])
        radius = 2.5

        # Generate exactly 4 points (minimal for sphere fitting)
        points = np.array([
            [1.0, 2.0, 5.5],  # point on sphere
            [3.5, 2.0, 3.0],  # point on sphere
            [1.0, 4.5, 3.0],  # point on sphere
            [-1.5, 2.0, 3.0]   # point on sphere
        ])

        sphere = Sphere()
        sphere.fit_model(points)

        # Check that the fitted parameters are close to the original
        assert np.allclose(sphere.center, center, atol=1e-6)
        assert np.isclose(sphere.radius, radius, atol=1e-6)

    def test_calc_distances(self, perfect_model: Sphere):
        # Test distance calculation with points on, inside, and outside the sphere
        center = perfect_model.center
        radius = perfect_model.radius

        # Points on the sphere (distance should be 0)
        on_sphere = center + radius * np.array([1.0, 0.0, 0.0])
        assert np.isclose(perfect_model.calc_distance_one_point(on_sphere), 0.0)

        # Points inside the sphere (distance should be positive)
        inside = center + 0.5 * radius * np.array([1.0, 0.0, 0.0])
        assert np.isclose(perfect_model.calc_distance_one_point(inside), 0.5 * radius)

        # Points outside the sphere (distance should be positive)
        outside = center + 1.5 * radius * np.array([1.0, 0.0, 0.0])
        assert np.isclose(perfect_model.calc_distance_one_point(outside), 0.5 * radius)

        # Test with multiple points
        points = np.array([on_sphere, inside, outside])
        distances = perfect_model.calc_distances(points)
        expected = np.array([0.0, 0.5 * radius, 0.5 * radius])
        assert np.allclose(distances, expected)

    def test_benchmark_starkit_ransac(
            self, 
            data_points, 
            benchmark
        ):
        ransac = RANSAC(data_points)
        benchmark(
                ransac.fit,
                Sphere,
                500, 
                0.1
        )

    def test_benchmark_pyransac(
            self, 
            data_points, 
            benchmark
        ):
        sphere = pyransac3d.Sphere()
        benchmark(
                sphere.fit,
                data_points,
                0.1,
                500, 
        )
