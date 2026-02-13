import pytest

from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.ellipsoid import Ellipsoid3D
import numpy as np

N_POINTS = 1000
NOISE_POINTS = 1000
SEED = 42
np.random.seed(SEED)

@pytest.fixture
def ellipsoid_parameters():
    ellipsoid = Ellipsoid3D()
    found = False
    while not found:
        minimum_points = np.random.random((9, 3))
        try:
            ellipsoid.fit_model(minimum_points)
        except ValueError as e:
            pass


@pytest.fixture
def acceptable_rmse():
    pass

def test_ellipsoid(ellipsoid_parameters):
    pass
