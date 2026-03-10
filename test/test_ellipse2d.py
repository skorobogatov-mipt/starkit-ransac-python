import pytest
import numpy as np

from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.ellipse2d import Ellipse2D
from starkit_ransac.generators.ellipse2d import generate_ellipse2d
from conftest import RNG


class TestEllipse2D:
    N_CENTERS = 5
    N_RADII = 5
    N_ROTATIONS = 5

    MAX_OFFSET = 20.0
    centers_list = (RNG.random((N_CENTERS, 2)) * MAX_OFFSET).tolist()

    MAX_RADIUS = 5.0
    radii_list = (np.abs(RNG.random((N_RADII, 2)) * MAX_RADIUS)).tolist()

    # Rotations in [0, 2π)
    rotations_list = (RNG.random(N_ROTATIONS) * 2 * np.pi).tolist()

    @pytest.fixture(scope="class", params=centers_list)
    def center(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=radii_list)
    def radii(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=rotations_list)
    def rotation(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def perfect_model(self, rotation, radii, center):
        """
        An Ellipse2D instance built directly from the known geometric parameters.
        This is the “ground truth” that we later try to recover.
        """
        return Ellipse2D(
            rotation=rotation,
            radius_1=radii[0],
            radius_2=radii[1],
            center=center,
        )

    @pytest.fixture(scope="class", params=[0.0, 0.01, 0.02, 0.05, 0.1])
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[5000, 2500, 1000, 500])
    def n_points(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def data_points(self, perfect_model, noise_sigma, n_points):
        return generate_ellipse2d(
            perfect_model,
            noise_sigma=noise_sigma,
            n_points=n_points,
        )

    @pytest.fixture(scope="class")
    def fit_model(self, data_points):
        ransac = RANSAC(data_points)
        model = ransac.fit(Ellipse2D, 5000, 0.05)
        return model

    @pytest.fixture()
    def acceptable_polynomial_relative_error(self):
        return 0.15

    @pytest.fixture()
    def acceptable_rotation_error(self):
        return 0.1

    @pytest.fixture()
    def acceptable_radii_relative_error(self):
        return 0.2

    @pytest.fixture()
    def acceptable_center_distance(self):
        return 0.25

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.5

    def test_polynomial_are_close(
        self,
        fit_model,
        perfect_model,
        acceptable_polynomial_relative_error,
    ):
        actual = perfect_model.polynomial
        fit = fit_model.polynomial

        # avoid zero values in polynomial
        denom = np.where(np.abs(actual) < 1e-12, 1e-12, np.abs(actual))
        diffs = np.abs(actual - fit) / denom
        assert (diffs < acceptable_polynomial_relative_error).all()

    def test_rotation_are_close(
        self,
        fit_model,
        perfect_model,
        acceptable_rotation_error,
    ):
        actual = perfect_model.rotation
        fit = fit_model.rotation

        diff = np.abs(actual - fit)
        diff = min(diff, 2 * np.pi - diff)

        assert diff < acceptable_rotation_error

    def test_radii_are_close(
        self,
        fit_model,
        perfect_model,
        acceptable_radii_relative_error,
    ):
        actual = np.array([perfect_model.major_radius, perfect_model.minor_radius])
        fit = np.array([fit_model.major_radius, fit_model.minor_radius])

        actual_sorted = np.sort(actual)
        fit_sorted = np.sort(fit)

        denom = np.where(actual_sorted == 0, 1e-12, actual_sorted)
        diffs = np.abs((actual_sorted - fit_sorted) / denom)

        assert (diffs < acceptable_radii_relative_error).all()

    def test_centers_are_close(
        self,
        fit_model,
        perfect_model,
        acceptable_center_distance,
    ):
        actual = perfect_model.center
        fit = fit_model.center
        dist = np.linalg.norm(actual - fit)

        assert dist < acceptable_center_distance

    def test_overall_close(
        self,
        perfect_model: Ellipse2D,
        fit_model: Ellipse2D,
        acceptable_point_rmse,
        n_points,
    ):
        perfect_points = generate_ellipse2d(
            perfect_model,
            n_points=n_points,
            noise_sigma=0.0,
        )

        distances = fit_model.calc_distances(perfect_points)
        rmse = np.sqrt(np.mean(distances ** 2))

        assert rmse < acceptable_point_rmse
