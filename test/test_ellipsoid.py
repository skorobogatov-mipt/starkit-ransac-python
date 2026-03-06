import pytest

from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D
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
        RNG.random((N_RADII, 3)) * MAX_RADIUS
    ).tolist()

    axes_list = RNG.random((N_NORMALS,3,3))
    # make sure that the third normal is perpendicular to the first two
    axes_list[:, 2] = np.cross(axes_list[:, 0], axes_list[:, 1])
    # make sure that the firs normal is orthogonal to the second two
    axes_list[:, 0] = np.cross(axes_list[:, 1], axes_list[:, 2])
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
            params=[0., 0.05, 0.1, 0.5]
    )
    def noise_sigma(self, request):
        return request.param

    @pytest.fixture(
            scope='class', 
            params=[5000, 2500, 1000, 500, 50]
    )
    def n_points(self, request):
        return request.param




