import numpy as np
from numpy.typing import NDArray


def generate_cylindrical_ring(
        inner_r,
        outer_r,
        height,
        center,
        n_points=1000,
        seed:int=42
    ):
    np.random.seed(seed)
    points = []
    for _ in range(n_points):
        radius = np.random.uniform(inner_r, outer_r)
        angle = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height / 2, height / 2)

        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        points.append([x, y, z])

    return np.array(points)

def generate_stairs(
    n_steps: int,
    step_height: float,
    step_width: float,
    stair_span: float,
    n_points: int,
    noise_sigma: float,
    rotation_deg: float,
) -> NDArray:
    # Generate treads + risers, rotate, add noise.
    theta = np.deg2rad(rotation_deg)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    points = []

    for i in range(n_steps):
        x = np.random.uniform(i * step_width, (i + 1) * step_width, n_points)
        y = np.random.uniform(0.0, stair_span, n_points)
        z = np.ones_like(x) * i * step_height
        points.append(np.column_stack((x, y, z)))

    for i in range(1, n_steps):
        x = np.ones(n_points) * i * step_width
        y = np.random.uniform(0.0, stair_span, n_points)
        z = np.random.uniform((i - 1) * step_height, i * step_height, n_points)
        points.append(np.column_stack((x, y, z)))

    cloud = np.vstack(points)
    cloud = cloud @ rot.T

    if noise_sigma > 0:
        cloud = cloud + np.random.normal(0.0, noise_sigma, cloud.shape)

    return cloud

def generate_ellipsoid_data(
        center,
        radii,
        rpy
    ):
    """Создает тестовые данные эллипсоида со случайными параметрами."""

    # Случайные параметры эллипсоида
    true_center = np.random.uniform(0.1, 1.0, 3)
    true_radii = np.random.uniform(0.05, 0.5, 3)

    # Случайные углы вращения
    angles = np.random.uniform(0, 2 * np.pi, 3)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    true_rotation = Rz @ Ry @ Rx

    points = []

    # Inliers (80% точек)
    n_inliers = int(N_POINTS * 0.8)
    for _ in range(n_inliers):
        # Случайная точка внутри единичной сферы
        while True:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1:
                break

        # Масштабируем до эллипсоида
        p_local = p * true_radii

        # Вращаем и сдвигаем
        p_global = true_rotation @ p_local + true_center

        # Добавляем небольшой шум
        p_global += np.random.normal(0, 0.02, 3)
        points.append(p_global)

    # Outliers (20% точек)
    n_outliers = N_POINTS - n_inliers
    for _ in range(n_outliers):
        p_global = np.random.uniform(0, 1.5, 3)
        points.append(p_global)

    data = np.array(points)

    return {
        'data': data,
        'true_center': true_center,
        'true_radii': true_radii,
        'true_rotation': true_rotation,
        'n_inliers': n_inliers
    }
