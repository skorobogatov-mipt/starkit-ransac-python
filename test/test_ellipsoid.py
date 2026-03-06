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

    normals_list = RNG.random((N_NORMALS,3,3))
    normals_list = normalize(normals_list, -1)
    nor
    


    @pytest.fixture(scope='class', params=centers_list)
    def center(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=radii_list)
    def radius(self, request):
        return np.array(request.param)

    @pytest.fixture(scope="class", params=normals)
    def normal(self, request):
        return request.param
