import numpy as np


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
