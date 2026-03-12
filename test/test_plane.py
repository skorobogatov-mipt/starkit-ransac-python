import pytest
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.plane import Plane3D
from starkit_ransac.generators.plane import generate_plane
from conftest import RNG

from pytest_benchmark.plugin import benchmark
import pyransac3d

class TestPlane3D:
    approx_center_list = [
        [10, 54.2, 100],
        [0, 0, 0],
        [-4.2, 5.76, -1.228]
    ]
    coeffs_list = [
        [1, 1, 1, 0],
        [0, 2, 5, 10],
        [-345, 0, 1235, 23.1],
        [-0.054, 123, 0, 482.5748],
        [-0.8, 0.9, 3.57, 5.423]
    ]

    @pytest.fixture(scope='class', params=coeffs_list)
    def coeffs(self, request):
        return np.array(request.param, float)

    @pytest.fixture(scope='class', params=approx_center_list)
    def approx_center(self, request):
        return np.array(request.param, float)

    @pytest.fixture(scope='class')
    def perfect_model(self, coeffs):
        return Plane3D(*coeffs)

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
    def data_points(self, perfect_model, noise_sigma, n_points, approx_center):
        return generate_plane(
            perfect_model,
            noise_sigma=noise_sigma,
            n_points=n_points,
            approx_center=approx_center,
            plane_size=1
        )

    @pytest.fixture(scope='class')
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(
            Plane3D,
            iter_num=1000,
            distance_threshold=0.1
        )
        return model

    @pytest.fixture()
    def acceptable_relative_coeff_error(self):
        return 0.1

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.2

    def test_coeffs_are_close(
            self, 
            perfect_model:Plane3D, 
            fit_model:Plane3D,
            acceptable_relative_coeff_error
        ):
        perfect = np.array([
            perfect_model.a, 
            perfect_model.b, 
            perfect_model.c, 
            perfect_model.d
        ])
        fit = np.array([
            fit_model.a, 
            fit_model.b, 
            fit_model.c, 
            fit_model.d
        ])
        # scale fit model
        scale_idx = np.argmax(np.abs(perfect))
        fit *= perfect[scale_idx]/fit[scale_idx]

        diffs = np.abs(fit - perfect)
        non_zero = np.logical_not(np.isclose(fit, 0))
        non_zero_diffs = diffs[non_zero] / perfect[non_zero]
        zero_diffs = diffs[np.logical_not(non_zero)]
        assert (non_zero_diffs < acceptable_relative_coeff_error).all()
        assert (zero_diffs < acceptable_relative_coeff_error).all()



    def test_overall_close(
            self,
            perfect_model: Plane3D,
            fit_model: Plane3D,
            acceptable_point_rmse
        ):
        # Generate points on the perfect line
        perfect_points = generate_plane(perfect_model, n_points=5000, noise_sigma=0)

        # Calculate distances from these points to the fit line
        distances = fit_model.calc_distances(perfect_points)
        rmse = np.sqrt(np.mean(np.sum(distances**2)))

        assert rmse < acceptable_point_rmse

    def test_benchmark_starkit_ransac(
            self, 
            data_points,
            benchmark
        ):
        ransac = RANSAC(data_points)
        benchmark(
            ransac.fit,
            Plane3D,
            500,
            0.1
        )

    def test_benchmark_pyransac(
            self, 
            data_points,
            benchmark
        ):
        line = pyransac3d.Plane()
        benchmark(
                line.fit,
                data_points,
                0.1,
                500
        )

