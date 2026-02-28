import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from starkit_ransac.surfaces.cylinder import Cylinder

N_POINTS = 1500
SEED = 42

@pytest.fixture
def cylinder_data():
    np.random.seed(SEED)
    radius = 0.8
    height = 2.0
    center = np.array([1.5, -1.0, 0.8])
    axis = np.array([0.2, 0.9, 0.4])
    axis = axis / np.linalg.norm(axis)
    noise_level = 0.02
    if np.allclose(axis, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(axis, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    points = []
    normals = []
    for _ in range(N_POINTS):
        theta = np.random.uniform(0, 2 * np.pi)
        h = np.random.uniform(-height / 2, height / 2)
        pt = center + h * axis + radius * (np.cos(theta) * u + np.sin(theta) * v)
        normal = np.cos(theta) * u + np.sin(theta) * v
        pt += noise_level * np.random.randn() * normal
        points.append(pt)
        normals.append(normal)
    return {
        'points': np.array(points),
        'normals': np.array(normals),
        'center': center,
        'axis': axis,
        'radius': radius,
        'height': height,
    }

@pytest.fixture
def acceptable_rmse_cylinder():
    return 0.1

@pytest.fixture
def acceptable_angle_error():
    return 0.1

@pytest.fixture
def acceptable_height_error():
    return 0.1

def perform_ransac_cylinder(
        points,
        normals,
        distance_threshold,
        max_iterations,
        min_inliers=None
    ):
    best_model = None
    best_inliers = 0
    n_points = len(points)
    if min_inliers is None:
        min_inliers = n_points // 2
    for _ in range(max_iterations):
        idx = np.random.choice(n_points, 2, replace=False)
        sample_points = points[idx]
        sample_normals = normals[idx]
        model = Cylinder()
        try:
            model.set_normals(sample_normals)
            model.fit_model(sample_points)
        except (ValueError, np.linalg.LinAlgError):
            continue
        distances = model.calc_distances(points)
        inliers_mask = distances < distance_threshold
        inlier_count = np.sum(inliers_mask)
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_model = model
    if best_model is not None and best_inliers >= min_inliers:
        return best_model
    return None

def _visualize_cylinder(
        points,
        model,
        true_center,
        true_axis,
        true_radius,
        true_height
    ):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='b', s=1, alpha=0.3, label='Point cloud')
    p0 = model.model['point_on_axis']
    axis = model.model['axis_direction']
    radius = model.model['radius']
    h_min = model.model['h_min']
    h_max = model.model['h_max']
    theta = np.linspace(0, 2 * np.pi, 30)
    h = np.linspace(h_min, h_max, 20)
    theta_grid, h_grid = np.meshgrid(theta, h)
    if np.allclose(axis, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(axis, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    X = p0[0] + h_grid * axis[0] + radius * (np.cos(theta_grid) * u[0] + np.sin(theta_grid) * v[0])
    Y = p0[1] + h_grid * axis[1] + radius * (np.cos(theta_grid) * u[1] + np.sin(theta_grid) * v[1])
    Z = p0[2] + h_grid * axis[2] + radius * (np.cos(theta_grid) * u[2] + np.sin(theta_grid) * v[2])
    ax.plot_surface(X, Y, Z, alpha=0.3, color='r', label='Fitted cylinder')
    t = np.linspace(-true_height/2, true_height/2, 10)
    ax.plot(true_center[0] + t * true_axis[0],
            true_center[1] + t * true_axis[1],
            true_center[2] + t * true_axis[2],
            'g--', linewidth=2, label='True axis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('RANSAC Cylinder Fitting')
    plt.show()

def test_cylinder(
        cylinder_data, 
        acceptable_rmse_cylinder,
        acceptable_angle_error, 
        acceptable_height_error
    ):
    points = cylinder_data['points']
    normals = cylinder_data['normals']
    model = perform_ransac_cylinder(
        points, normals,
        distance_threshold=0.05,
        max_iterations=1000,
        min_inliers=N_POINTS // 2
    )
    assert model is not None
    model.set_height_from_points(points)
    result = model.get_model()
    assert abs(result['radius'] - cylinder_data['radius']) < acceptable_rmse_cylinder
    axis_result = result['axis_direction']
    axis_gt = cylinder_data['axis']
    angle_diff = min(
        np.arccos(np.clip(np.abs(np.dot(axis_result, axis_gt)), -1, 1)),
        np.arccos(np.clip(np.abs(np.dot(-axis_result, axis_gt)), -1, 1))
    )
    assert angle_diff < acceptable_angle_error
    point_on_axis = result['point_on_axis']
    proj_center = point_on_axis + np.dot(cylinder_data['center'] - point_on_axis, axis_result) * axis_result
    center_error = np.linalg.norm(proj_center - cylinder_data['center'])
    assert center_error < acceptable_rmse_cylinder * 2
    assert abs(result['h_max'] - result['h_min'] - cylinder_data['height']) < acceptable_height_error
    distances = model.calc_distances(points)
    assert np.percentile(distances, 90) < 0.1
    if os.environ.get('VISUALIZE_TESTS'):
        _visualize_cylinder(points, model,
                            cylinder_data['center'],
                            cylinder_data['axis'],
                            cylinder_data['radius'],
                            cylinder_data['height'])

def test_cylinder_no_normals(cylinder_data):
    points = cylinder_data['points']
    idx = np.random.choice(len(points), 2, replace=False)
    model = Cylinder()
    with pytest.raises(ValueError, match="Normals must be provided"):
        model.fit_model(points[idx])

def test_cylinder_insufficient_points():
    points = np.random.random((1, 3))
    normals = np.random.random((1, 3))
    model = Cylinder()
    model.set_normals(normals)
    with pytest.raises(AssertionError):
        model.fit_model(points)

def test_cylinder_distance_calculation(cylinder_data):
    points = cylinder_data['points']
    normals = cylinder_data['normals']
    model = perform_ransac_cylinder(
        points, normals,
        distance_threshold=0.05,
        max_iterations=500,
        min_inliers=N_POINTS // 2
    )
    assert model is not None
    model.set_height_from_points(points)
    surface_points = points[:100]
    distances = model.calc_distances(surface_points)
    assert np.mean(distances) < 0.1
    far_point = np.array([[10, 10, 10]])
    far_distance = model.calc_distances(far_point)
    assert far_distance[0] > 5.0

def test_cylinder_height_setting(cylinder_data):
    points = cylinder_data['points']
    normals = cylinder_data['normals']
    model = perform_ransac_cylinder(
        points, normals,
        distance_threshold=0.05,
        max_iterations=500,
        min_inliers=N_POINTS // 2
    )
    assert model is not None
    margin = 0.1
    model.set_height_from_points(points, margin=margin)
    new_height = model.model['h_max'] - model.model['h_min']
    assert abs(new_height - cylinder_data['height'] - 2*margin) < 0.1

def test_cylinder_model_serialization(cylinder_data):
    points = cylinder_data['points']
    normals = cylinder_data['normals']
    model = perform_ransac_cylinder(
        points, normals,
        distance_threshold=0.05,
        max_iterations=500,
        min_inliers=N_POINTS // 2
    )
    assert model is not None
    result = model.get_model()
    assert 'point_on_axis' in result
    assert 'axis_direction' in result
    assert 'radius' in result
    assert 'h_min' in result
    assert 'h_max' in result
    assert isinstance(result['radius'], float)
    assert isinstance(result['point_on_axis'], np.ndarray)
    assert result['point_on_axis'].shape == (3,)
    
