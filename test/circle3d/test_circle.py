import pyransac3d
import numpy as np
import pytest
from starkit_ransac.generators.circle import generate_circle
from starkit_ransac.surfaces.circle import Circle3D
from starkit_ransac.ransac_3d import RANSAC
from conftest import BENCHMARK_THRESH, N_ITER_BENCHMARK, SEED, RNG, AbstractTestPrecision
from pytest_benchmark.plugin import benchmark

from circle3d_generator import Circle3DGenerator

from pyransac3d import Circle

class TestCircle3D(Circle3DGenerator):
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
        self, fitted_circle, perfect_circle, acceptable_radius_error
    ):
        fit_radius = fitted_circle.radius
        actual_radius = perfect_circle.radius
        relative_radius_error = abs(fit_radius - actual_radius) / actual_radius
        assert relative_radius_error < acceptable_radius_error

    def test_normals_are_close(
        self, fitted_circle, perfect_circle, acceptable_normal_error
    ):
        fit_normal = fitted_circle.normal
        actual_normal = perfect_circle.normal
        dist1 = np.linalg.norm(fit_normal - actual_normal)
        dist2 = np.linalg.norm(fit_normal + actual_normal)
        assert min(dist1, dist2) < acceptable_normal_error

    def test_centers_are_close(
        self, fitted_circle, perfect_circle, acceptable_center_error
    ):
        fit_center = fitted_circle.center
        actual_center = perfect_circle.center
        dist = np.linalg.norm(fit_center - actual_center)
        assert dist < acceptable_center_error


class TestBenchmarkCircle3D(Circle3DGenerator):

    def test_benchmark_starkit_ransac(self, circle_data, benchmark):
        ransac = RANSAC(circle_data)
        benchmark(ransac.fit, Circle3D, N_ITER_BENCHMARK, BENCHMARK_THRESH)

    def test_benchmark_pyransac(self, circle_data, benchmark):
        circle = pyransac3d.circle.Circle()
        benchmark(circle.fit, circle_data, BENCHMARK_THRESH, N_ITER_BENCHMARK)

# class TestPrecisionCircle3D(Circle3DGenerator, AbstractTestPrecision):
#
#     def test_compare_rmse(self, perfect_model, data_points, n_iter):
#         all_stransac_distances = 0
#         all_pyransac_distances = 0
#         perfect_data = generate_circle(perfect_model, 0, 1000)
#         pyransac_circle = Circle()
#         for i in range(self.N_ITER_AVERAGE):
#             ransac = RANSAC(data_points)
#             fit_circle = ransac.fit(Circle3D, n_iter, self.THRESHOLD)
#             stransac_distance = np.mean(fit_circle.calc_distances(perfect_data))
#             all_stransac_distances += stransac_distance
#
#             center, axis, radius, _ = pyransac_circle.fit(data_points, self.THRESHOLD, n_iter)
#
#             circle_from_pyransac = Circle3D(center, float(radius), axis)
#             pyransac_distances = circle_from_pyransac.calc_distances(perfect_data)
#             pyransac_rmse = np.mean(pyransac_distances)
#             all_pyransac_distances += pyransac_rmse
#
#         starkit_ransac_avg_distance = all_stransac_distances / n_iter
#         pyransac_avg_distance = all_pyransac_distances / n_iter
#         print('starkit RMSE: ', starkit_ransac_avg_distance)
#         print('pyransac RMSE: ', pyransac_avg_distance)
#         self._append_result(
#                 self.generate_result_dict(
#                     n_iter, 
#                     starkit_ransac_avg_distance,
#                     pyransac_avg_distance,
#                     len(data_points),
#                     self.SHAPE_NAME
#                 )
#         )
