import pytest

from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.plane import Plane3D
import numpy as np

N_POINTS = 50
SEED = 42

@pytest.fixture
def point_data():
    np.random.seed(SEED)

    a, b, c = np.array((0,0,1), dtype=float)
    norm = np.sqrt(a**2 + b**2 + c**2)
    a, b, c = a/norm, b/norm, c/norm
    
    x = np.random.uniform(-4, 4, N_POINTS)
    y = np.random.uniform(-4, 4, N_POINTS)
    
    # Вычисляем z из уравнения плоскости
    x0, y0, z0 = (0,0,0)
    d = - (a*x0 + b*y0 + c*z0)
    z = (-d - a*x - b*y) / c
    
    # Добавляем шум
    noise = 0.1
    noise_vec = np.random.randn(N_POINTS) * noise
    z += noise_vec

    return np.random.random((N_POINTS, 3))

@pytest.fixture
def acceptable_rmse():
    return 0.2
    
def test_plane(point_data, acceptable_rmse):
    runsuck = RANSAC3D()
    runsuck.add_points(point_data)

    model = runsuck.fit(
            Plane3D,
            1000,
            0.5
    )
    result = model.get_model()
    result_point = np.array([
        result['a'], 
        result['b'], 
        result['c'], 
        result['d']
    ])

    data_mean = np.array([0,0,1,0])
    assert np.linalg.norm(result_point - data_mean) < acceptable_rmse
