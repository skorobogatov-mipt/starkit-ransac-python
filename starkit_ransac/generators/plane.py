from starkit_ransac.surfaces.plane import Plane3D
import numpy as np

def generate_plane(
        plane:Plane3D,
        noise_sigma=0.05,
        num_points=1000, 
        plane_size=10
    ):
    points = []
    a = plane.a
    b = plane.b
    c = plane.c
    d = plane.d

    x_range = (-plane_size/2, plane_size/2)
    y_range = (-plane_size/2, plane_size/2)
    if c == 0:
        if b == 0:
            for _ in range(num_points):
                y = np.random.uniform(*y_range)
                z = np.random.uniform(*y_range)
                x = (d - b*y - c*z) / a
                points.append((x, y, z))
        else:
            for _ in range(num_points):
                x = np.random.uniform(*x_range)
                z = np.random.uniform(*y_range)
                y = (d - a*x - c*z) / b
                points.append((x, y, z))
    else:
        for _ in range(num_points):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = (d - a*x - b*y) / c
            points.append((x, y, z))

    points = np.array(points)
    points += np.random.normal(0, noise_sigma, points.shape)
    return points
