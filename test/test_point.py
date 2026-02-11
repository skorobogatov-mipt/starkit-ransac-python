import pytest

from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.point import Point3D
import numpy as np

N_POINTS = 1000
SEED = 42

@pytest.fixture
def point_data():
    np.random.seed(SEED)
    return np.random.random((N_POINTS, 3))

@pytest.fixture
def acceptable_rmse():
    return 0.2
    
def test_point(point_data, acceptable_rmse):
    runsuck = RANSAC3D()
    runsuck.add_points(point_data)

    model = runsuck.fit(
            Point3D,
            1000,
            0.5
    )
    result = model.get_model()
    result_point = np.array([
        result['x'], 
        result['y'], 
        result['z']
    ])

    data_mean = np.mean(point_data, axis=0)
    assert np.linalg.norm(result_point - data_mean) < acceptable_rmse

