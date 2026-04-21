from pyransac3d.line import Line
import pytest
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.line3d import Line3D
from starkit_ransac.generators.line3d import generate_line3d
from conftest import BENCHMARK_THRESH, N_ITER_BENCHMARK, RNG
from tqdm import tqdm
from filelock import FileLock

from pytest_benchmark.plugin import benchmark

import pyransac3d
from line_generator import LineGenerator

import json
import os

class TestLine3D(LineGenerator):
    @pytest.fixture()
    def acceptable_direction_rmse(self):
        return 0.1

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.2

    @pytest.fixture()
    def direction_rmse(self, fit_model, perfect_model):
        actual = perfect_model.direction
        fit = fit_model.direction

        # Directions can be in opposite directions, so we check both
        pos_diff = np.linalg.norm(actual - fit)
        neg_diff = np.linalg.norm(actual + fit)
        min_diff = min(pos_diff, neg_diff)
        return min_diff

    @pytest.fixture()
    def point_rmse(self, fit_model, perfect_model):
        # Project the perfect model's point onto the fit model's line
        # to get the closest point on the fit line
        a = fit_model.point
        v = fit_model.direction
        p = perfect_model.point

        # Calculate the projection of (p - a) onto v
        projection = np.dot(p - a, v) * v
        closest_point = a + projection

        # Calculate distance between perfect point and closest point on fit line
        distance = np.linalg.norm(p - closest_point)
        return distance

    @pytest.fixture()
    def overall_point_distance(self, fit_model, perfect_model):
        # Generate points on the perfect line
        perfect_points = generate_line3d(perfect_model, n_points=5000, noise_sigma=0)

        # Calculate distances from these points to the fit line
        distances = fit_model.calc_distances(perfect_points)
        rmse = np.sqrt(np.mean(np.sum(distances**2, axis=1)))
        return rmse

    def test_direction_is_close(
        self, direction_rmse, acceptable_direction_rmse
    ):
        assert direction_rmse < acceptable_direction_rmse

    def test_point_is_close(self, point_rmse, acceptable_point_rmse):
        assert point_rmse < acceptable_point_rmse

    def test_overall_close(
        self, overall_point_distance, acceptable_point_rmse
    ):
        assert overall_point_distance < acceptable_point_rmse

    def test_fit_model_method(self):
        # Test the fit_model method directly
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

        line = Line3D()
        line.fit_model(points)

        # Check if direction is along x-axis
        assert np.allclose(line.direction, [1, 0, 0])

        # Check if point is at origin
        assert np.allclose(line.point, [0, 0, 0])

    def test_calc_distances_method(self):
        # Test the calc_distances method
        line = Line3D(direction=[1, 0, 0], point=[0, 0, 0])

        # Points on the line should have zero distance
        points_on_line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        distances = line.calc_distances(points_on_line)
        assert np.allclose(distances, 0)

        # Points not on the line should have non-zero distance
        points_off_line = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]])
        distances = line.calc_distances(points_off_line)
        assert np.allclose(np.linalg.norm(distances, axis=1), 1)


class TestBenchmarkLine3D(LineGenerator):

    def test_benchmark_starkit_ransac(self, data_points, benchmark):
        ransac = RANSAC(data_points)
        benchmark(ransac.fit, Line3D, N_ITER_BENCHMARK, BENCHMARK_THRESH)

    def test_benchmark_pyransac(self, data_points, benchmark):
        line = pyransac3d.Line()
        benchmark(line.fit, data_points, BENCHMARK_THRESH, N_ITER_BENCHMARK)

class TestPrecisionLine3D(LineGenerator):
    n_iter_list = [500, 1000, 2000, 3000]
    THRESHOLD = 0.1
    N_ITER_AVERAGE = 10

    @pytest.fixture(scope="class", params=n_iter_list)
    def n_iter(self, request):
        return request.param
    
    RESULT_PATH = 'precision_comparison.json'
    LOCK_PATH = RESULT_PATH + '.lock'

    @staticmethod
    def _append_result(result):
        with FileLock(TestPrecisionLine3D.LOCK_PATH):
            path = TestPrecisionLine3D.RESULT_PATH
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            data.append(result)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    def test_compare_rmse(self, perfect_model, data_points, n_iter):
        all_stransac_distances = 0
        all_pyransac_distances = 0
        perfect_data = generate_line3d(perfect_model, 0, 1000)
        pyransac_line = Line()
        for i in range(self.N_ITER_AVERAGE):
            ransac = RANSAC(data_points)
            fit_line = ransac.fit(Line3D, n_iter, self.THRESHOLD)
            stransac_distance = np.mean(fit_line.calc_distances(perfect_data))
            all_stransac_distances += stransac_distance

            A, B, _ = pyransac_line.fit(data_points, self.THRESHOLD, n_iter)
            A = np.array(A)
            B = np.array(B)
            A = A / np.linalg.norm(A)
            vecC_stakado = np.stack([A] * len(perfect_data), 0)
            dist_pt = np.cross(vecC_stakado, (B - perfect_data))
            dist_pt = np.linalg.norm(dist_pt, axis=1)
            pyransac_distance = np.mean(dist_pt)
            all_pyransac_distances += pyransac_distance

        starkit_ransac_avg_distance = all_stransac_distances / n_iter
        pyransac_avg_distance = all_pyransac_distances / n_iter
        print('starkit RMSE: ', starkit_ransac_avg_distance)
        print('pyransac RMSE: ', pyransac_avg_distance)
        self._append_result({
            'n_iter' : n_iter,
            'starkit_ransac RMSE' : starkit_ransac_avg_distance,
            'pyransac RMSE' : pyransac_avg_distance,
            'n data' : len(data_points)
        })

