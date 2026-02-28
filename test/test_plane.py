import pytest

from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.plane import Plane3D
import numpy as np

N_POINTS = 100
SEED = 42

@pytest.fixture
def plane_data():
    np.random.seed(SEED)

    a, b, c = np.array((0,0,1), dtype=float)
    norm = np.sqrt(a**2 + b**2 + c**2)
    a, b, c = a/norm, b/norm, c/norm
    print(a,b,c)
    
    x = np.random.uniform(-10, 10, N_POINTS)
    y = np.random.uniform(-10, 10, N_POINTS)
    
    # Вычисляем z из уравнения плоскости
    x0, y0, z0 = (0,0,0)
    d = - (a*x0 + b*y0 + c*z0)
    z = (-d - a*x - b*y) / c
    
    # Добавляем шум
    noise = 0.01
    noise_vec = np.random.uniform(-noise, noise, N_POINTS)
    z += noise_vec
    points = np.vstack((x,y,z)).T
    return points

@pytest.fixture
def acceptable_rmse():
    return 0.2
    
def test_plane(plane_data, acceptable_rmse):
    runsuck = RANSAC3D()
    runsuck.add_points(plane_data)

    model = runsuck.fit(
            Plane3D,
            1000,
            0.4
    )
    result = model.get_model()
    result_plane= np.array([
        result['a'], 
        result['b'], 
        result['c'], 
        result['d']
    ])

    data_target = np.array([0,
                            0,
                            1,
                            0])
    vec_mul = np.array([result_plane[1]*data_target[2] - result_plane[2]*data_target[1],
                       result_plane[2]*data_target[0] - result_plane[0]*data_target[2],
                       result_plane[0]*data_target[1] - result_plane[1]*data_target[0]])
    assert np.linalg.norm(vec_mul) + result_plane[3]-data_target[3] < acceptable_rmse
