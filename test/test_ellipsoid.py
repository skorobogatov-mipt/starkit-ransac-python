import pytest

from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D
from starkit_ransac.generators.ellipsoid import generate_ellipsoid
import numpy as np

from conftest import RNG
from starkit_ransac.utils import normalize

class TestEllipsoid3D:
    # numbers of variants of a parameter
    N_CENTERS = 5
    N_RADII = 5
    N_NORMALS = 5

    MAX_OFFSET = 20
    centers_list = (
        RNG.random((N_CENTERS, 3)) * MAX_OFFSET
    ).tolist()

    MAX_RADIUS = 5
    radii_list = (
        np.abs(RNG.random((N_RADII, 3)) * MAX_RADIUS)
    ).tolist()

    axes_list = RNG.random((N_NORMALS,3,3))
    # make sure that the third normal is perpendicular to the first two
    axes_list[:, 2] = np.cross(axes_list[:, 0], axes_list[:, 1])
    # make sure that the firs normal is orthogonal to the second two axes_list[:, 0] = np.cross(axes_list[:, 1], axes_list[:, 2])
    axes_list[:, 1] = np.cross(axes_list[:, 2], axes_list[:, 0])
    axes_list = normalize(axes_list, axis=-1).tolist()


    @pytest.fixture(scope='class', params=centers_list)
    def center(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=radii_list)
    def radii(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=axes_list)
    def axes(self, request):
        return request.param

    @pytest.fixture(scope='class')
    def perfect_model(
            self,
            axes,
            radii,
            center
        ):
        return Ellipsoid3D(axes, radii, center)

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
        return generate_ellipsoid(
            perfect_model,
            noise_sigma=noise_sigma,
            n_points=n_points
        )

    @pytest.fixture(scope='class')
    def fit_model(self, data_points):
        rasnsac = RANSAC(data_points)
        model = rasnsac.fit(
            Ellipsoid3D,
            10000,
            0.1
        )

        return model

    @pytest.fixture()
    def acceptable_polynomial_relative_error(self):
        return 0.1

    @pytest.fixture()
    def acceptable_axes_rmse(self):
        return 0.2

    @pytest.fixture()
    def acceptable_radii_relative_error(self):
        return 0.2

    @pytest.fixture()
    def acceptable_center_distance(self):
        return 0.25

    def test_polynomial_are_close(self, fit_model, perfect_model, acceptable_polynomial_relative_error):
        actual = perfect_model.model['polynomial']
        fit = fit_model.model['polynomial']
        diffs = np.abs((actual - fit))/actual
        assert (diffs < acceptable_polynomial_relative_error).all()

    def test_axes_are_close(self, fit_model, perfect_model, acceptable_axes_rmse):
        actual = perfect_model.model['axes']
        fit = fit_model.model['axes']
        # axes can be in any order, so we have to find the minimal difference
        diffs = []

        # some axes may be inverted, which cnanges nothing
        for ax_fit, ax_axtual in zip(actual, fit):
            pos_diff = np.linalg.norm(ax_fit - ax_axtual)
            neg_diff = np.linalg.norm(ax_fit + ax_axtual)
            diffs.append(min(pos_diff, neg_diff))
            
        # if np.linalg.norm(diffs) > 1:
        #     pdb.set_trace()
        assert np.linalg.norm(diffs) < acceptable_axes_rmse

    def test_radii_are_close(self, fit_model, perfect_model, acceptable_radii_relative_error):
        actual = perfect_model.model['radii']
        fit = fit_model.model['radii']

        # sort them because the order can be different
        fit = np.sort(fit)
        actual = np.sort(actual)

        diffs = np.abs((actual - fit) / actual)
        assert (diffs < acceptable_radii_relative_error).all()

    def test_centers_are_close(
            self, 
            fit_model, 
            perfect_model, 
            acceptable_center_distance
        ):
        actual = perfect_model.model['center']
        fit = fit_model.model['center']
        diffs = actual - fit
        assert np.linalg.norm(diffs) < acceptable_center_distance

    @pytest.fixture()
    def acceptable_point_rmse(self):
        return 0.5

    def test_overall_close(
            self,
            perfect_model:Ellipsoid3D,
            fit_model:Ellipsoid3D,
            acceptable_point_rmse
        ):
        perfect_points = generate_ellipsoid(perfect_model, n_points=5000, noise_sigma=0)
        distances = fit_model.calc_distances(perfect_points)
        rmse = np.sqrt(np.mean(distances**2))
        assert rmse < acceptable_point_rmse


