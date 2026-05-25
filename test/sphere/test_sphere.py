from pyransac3d import Sphere as PyransacSphere
import pytest
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.sphere import Sphere
from starkit_ransac.generators.sphere import generate_sphere
from conftest import BENCHMARK_THRESH, N_ITER_BENCHMARK, AbstractTestPrecision

from pytest_benchmark.plugin import benchmark

import pyransac3d
import scuf
from sphere_generator import SphereGenerator


class TestSphere(SphereGenerator):
    @pytest.fixture()
    def acceptable_radius_relative_error(self):
        return 0.1

    @pytest.fixture()
    def acceptable_center_distance(self):
        return 0.25

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.5

    def test_radius_is_close(
        self, fit_model, perfect_model, acceptable_radius_relative_error
    ):
        actual = perfect_model.radius
        fit = fit_model.radius
        diff = np.abs((actual - fit) / actual)
        assert diff < acceptable_radius_relative_error

    def test_center_is_close(
        self, fit_model, perfect_model, acceptable_center_distance
    ):
        actual = perfect_model.center
        fit = fit_model.center
        diff = np.linalg.norm(actual - fit)
        assert diff < acceptable_center_distance

    def test_overall_close(
        self, perfect_model: Sphere, fit_model: Sphere, acceptable_point_rmse
    ):
        # Generate points on the perfect sphere
        n_points = 5000
        points = generate_sphere(perfect_model, n_points=n_points, noise_sigma=0)

        # Calculate distances to the fitted sphere
        distances = fit_model.calc_distances(points)
        rmse = np.sqrt(np.mean(distances**2))
        assert rmse < acceptable_point_rmse

    def test_fit_model_with_minimal_points(self):
        # Test that the model can be fit with the minimal number of points
        center = np.array([1.0, 2.0, 3.0])
        radius = 2.5

        # Generate exactly 4 points (minimal for sphere fitting)
        points = np.array(
            [
                [1.0, 2.0, 5.5],  # point on sphere
                [3.5, 2.0, 3.0],  # point on sphere
                [1.0, 4.5, 3.0],  # point on sphere
                [-1.5, 2.0, 3.0],  # point on sphere
            ]
        )

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


class TestBenchmarkSphere(SphereGenerator):

    def test_benchmark_starkit_ransac(self, data_points, benchmark):
        ransac = RANSAC(data_points)
        benchmark(ransac.fit, Sphere, N_ITER_BENCHMARK, BENCHMARK_THRESH)

    def test_benchmark_pyransac(self, data_points, benchmark):
        sphere = pyransac3d.Sphere()
        benchmark(sphere.fit, data_points, BENCHMARK_THRESH, N_ITER_BENCHMARK)

    def test_benchmark_scuf(self, data_points, benchmark):
        rs = scuf.ransac.RANSAC(figure="ellipsoid")
        benchmark(
            rs.fit, data_points, iterations=N_ITER_BENCHMARK, threshold=BENCHMARK_THRESH
        )


class TestPrecisionSphere(SphereGenerator, AbstractTestPrecision):

    @staticmethod
    def _scuf_ellipsoid_distances(points, center, radii, rotation):
        # Transform points into the ellipsoid's principal frame
        local = (points - center) @ rotation.T
        norms = np.linalg.norm(local, axis=-1)
        # Avoid division by zero for points exactly at the center
        safe = np.where(norms > 0, norms, 1.0)
        directions = local / safe[:, np.newaxis]
        # Distance from center to ellipsoid surface along each direction
        t = 1.0 / np.sqrt(np.sum((directions / radii) ** 2, axis=-1))
        return np.abs(norms - t)

    def test_compare_rmse(self, perfect_model, data_points, n_iter):
        all_stransac_distances = 0
        all_pyransac_distances = 0
        all_scuf_distances = 0
        perfect_data = generate_sphere(perfect_model, n_points=1000, noise_sigma=0)
        pyransac_sphere = PyransacSphere()
        scuf_ransac = scuf.ransac.RANSAC(figure='ellipsoid')
        for i in range(self.N_ITER_AVERAGE):
            ransac = RANSAC(data_points)
            fit_sphere = ransac.fit(Sphere, n_iter, self.THRESHOLD)
            stransac_distance = np.mean(fit_sphere.calc_distances(perfect_data))
            all_stransac_distances += stransac_distance

            center, radius, _ = pyransac_sphere.fit(
                data_points, self.THRESHOLD, n_iter
            )
            center = np.array(center)
            dist_pt = np.abs(
                np.linalg.norm(perfect_data - center, axis=-1) - radius
            )
            pyransac_distance = np.mean(dist_pt)
            all_pyransac_distances += pyransac_distance

            _, (scuf_center, scuf_radii, scuf_rotation) = scuf_ransac.fit(
                data_points, iterations=n_iter, threshold=self.THRESHOLD
            )
            scuf_dist_pt = self._scuf_ellipsoid_distances(
                perfect_data,
                np.array(scuf_center),
                np.array(scuf_radii),
                np.array(scuf_rotation),
            )
            scuf_distance = np.mean(scuf_dist_pt)
            all_scuf_distances += scuf_distance

        starkit_ransac_avg_distance = all_stransac_distances / n_iter
        pyransac_avg_distance = all_pyransac_distances / n_iter
        scuf_avg_distance = all_scuf_distances / n_iter
        print('starkit RMSE: ', starkit_ransac_avg_distance)
        print('pyransac RMSE: ', pyransac_avg_distance)
        print('scuf RMSE: ', scuf_avg_distance)
        self._append_result(
            self.generate_result_dict(
                n_iter,
                starkit_ransac_avg_distance,
                pyransac_avg_distance,
                len(data_points),
                self.SHAPE_NAME,
                scuf_rmse=scuf_avg_distance,
            )
        )
