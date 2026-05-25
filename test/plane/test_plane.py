from pyransac3d.plane import Plane
import pytest
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.plane import Plane3D
from starkit_ransac.generators.plane import generate_plane
from conftest import BENCHMARK_THRESH, N_ITER_BENCHMARK, AbstractTestPrecision

from pytest_benchmark.plugin import benchmark

import pyransac3d
import scuf
from plane_generator import PlaneGenerator


class TestPlane3D(PlaneGenerator):
    @pytest.fixture()
    def acceptable_relative_coeff_error(self):
        return 0.1

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.2

    def test_coeffs_are_close(
        self,
        perfect_model: Plane3D,
        fit_model: Plane3D,
        acceptable_relative_coeff_error,
    ):
        perfect = np.array(
            [perfect_model.a, perfect_model.b, perfect_model.c, perfect_model.d]
        )
        fit = np.array([fit_model.a, fit_model.b, fit_model.c, fit_model.d])
        # scale fit model
        scale_idx = np.argmax(np.abs(perfect))
        fit *= perfect[scale_idx] / fit[scale_idx]

        diffs = np.abs(fit - perfect)
        non_zero = np.logical_not(np.isclose(fit, 0))
        non_zero_diffs = diffs[non_zero] / perfect[non_zero]
        zero_diffs = diffs[np.logical_not(non_zero)]
        assert (non_zero_diffs < acceptable_relative_coeff_error).all()
        assert (zero_diffs < acceptable_relative_coeff_error).all()

    def test_overall_close(
        self, perfect_model: Plane3D, fit_model: Plane3D, acceptable_point_rmse
    ):
        # Generate points on the perfect line
        perfect_points = generate_plane(perfect_model, n_points=5000, noise_sigma=0)

        # Calculate distances from these points to the fit line
        distances = fit_model.calc_distances(perfect_points)
        rmse = np.sqrt(np.mean(np.sum(distances**2)))

        assert rmse < acceptable_point_rmse

class TestBenchmarkPlane3D(PlaneGenerator):

    def test_benchmark_starkit_ransac(self, data_points, benchmark):
        ransac = RANSAC(data_points)
        benchmark(ransac.fit, Plane3D, N_ITER_BENCHMARK, BENCHMARK_THRESH)

    def test_benchmark_pyransac(self, data_points, benchmark):
        line = pyransac3d.Plane()
        benchmark(line.fit, data_points, BENCHMARK_THRESH, N_ITER_BENCHMARK)

class TestPrecisionPlane3D(PlaneGenerator, AbstractTestPrecision):
    def test_compare_rmse(self, perfect_model, data_points, n_iter):
        all_stransac_distances = 0
        all_pyransac_distances = 0
        perfect_data = generate_plane(perfect_model, 0, 1000)
        pyransac_plane = Plane()
        for i in range(self.N_ITER_AVERAGE):
            ransac = RANSAC(data_points)
            fit_plane = ransac.fit(Plane3D, n_iter, self.THRESHOLD)
            stransac_distance = np.mean(fit_plane.calc_distances(perfect_data))
            all_stransac_distances += stransac_distance

            (A, B, C, D) , _ = pyransac_plane.fit(data_points, self.THRESHOLD, n_iter)

            plane_from_pyransac = Plane3D(A, B, C, D)
            pyransac_distances = plane_from_pyransac.calc_distances(perfect_data)
            pyransac_rmse = np.mean(pyransac_distances)
            all_pyransac_distances += pyransac_rmse

        starkit_ransac_avg_distance = all_stransac_distances / n_iter
        pyransac_avg_distance = all_pyransac_distances / n_iter
        print('starkit RMSE: ', starkit_ransac_avg_distance)
        print('pyransac RMSE: ', pyransac_avg_distance)
        self._append_result(
                self.generate_result_dict(
                    n_iter, 
                    starkit_ransac_avg_distance,
                    pyransac_avg_distance,
                    len(data_points),
                    self.SHAPE_NAME
                )
        )
