import pytest

from ransac3d.ransac_3d import RANSAC3D
from ransac3d.surfaces.Mobius_strip import Mobius_strip
import numpy as np

N_ITERATIONS = 200
N_POINTS = 200
N_UNIFORM_POINTS=0
SEED = 42

@pytest.fixture
def test_generator_mobius_strip():
    test_info_points = np.array([
        [9, 9, 9],
        [1, 6, 7],
        [7, 0, 11],
        [5, 5, 10]
    ])
    test_orientation = 1
    test_mobius = Mobius_strip()
    test_mobius.fit_model(test_info_points)
    return test_mobius

@pytest.fixture
def point_data():
    np.random.seed(SEED)
    test_info_points = np.array([
        [9, 9, 9],
        [1, 6, 7],
        [7, 0, 11],
        [5, 5, 10]
    ])
    test_orientation = 1
    test_mobius = Mobius_strip()
    test_mobius.fit_model(test_info_points)
    
    test_noise_std = 0.5
    test_box_center = test_mobius.model['center']
    test_box_size = 10

    width = test_mobius.model['width']
    center = test_mobius.model['center']
    normal = test_mobius.model['normal']
    radius = test_mobius.model['radius']
    start_vector = test_mobius.model['start_vector']
    orientation = test_mobius.model['orientation']

    half_width = width/2
    v1 = np.cross(normal, start_vector)
    v1 /= np.linalg.norm(v1)
    angles = np.linspace(0, 2*np.pi, N_POINTS, endpoint=False)
    values = np.random.uniform(0, 1, size=N_POINTS)
    mobius_points = np.zeros((N_POINTS, 3))
    noise = np.random.normal(loc=0.0, scale=test_noise_std, size=mobius_points.shape)
    for i in range(N_POINTS):
        vector_in_circle_plane = np.cos(angles[i]) * start_vector + np.sin(angles[i]) * v1
        p = radius * vector_in_circle_plane
        half_width_vector = half_width*(vector_in_circle_plane * 
                                        np.cos(angles[i]*orientation/2) +
                                        normal*np.sin(angles[i]*orientation/2))
        one_end = p+half_width_vector
        another_end = p-half_width_vector
        mobius_points[i, :] = another_end * values[i] + one_end * (1-values[i]) + center

    if N_UNIFORM_POINTS >0:
        points_uniform = test_box_center + np.random.uniform(
        low=-test_box_size / 2,
        high= test_box_size / 2,
        size=(N_UNIFORM_POINTS, 3)
        )
        return np.vstack([mobius_points+noise, points_uniform])
    return mobius_points+noise

@pytest.fixture
def acceptable_rmse():
    return 2
    
def test_mobius_strip(point_data, acceptable_rmse, test_generator_mobius_strip):
    runsuck = RANSAC3D()
    runsuck.add_points(point_data)

    model = runsuck.fit(
            Mobius_strip,
            N_ITERATIONS,
            0.5
    )
    result = model.get_model()
    result_center = result['center']
    result_radius = result['radius']
    result_width = result['width']

    assert((np.sum((result_center - test_generator_mobius_strip.model['center'])**2) +
            (result_radius - test_generator_mobius_strip.model['radius'])**2 +
            (result_width - test_generator_mobius_strip.model['width'])**2)
            /5 <  acceptable_rmse)

