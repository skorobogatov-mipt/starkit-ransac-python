import numpy as np
from starkit_ransac.surfaces.circle import Circle
from starkit_ransac.utils import rotate_from_axis_to_axis

def generate_circle(
        circle:Circle, 
        noise_sigma:float=0.05,
        n_points=1000
    ):
    center = circle.center
    normal = circle.normal
    radius = circle.radius

    # 1) generate a flat circle
    angles = np.linspace(-np.pi, np.pi, n_points)
    points = np.vstack([
        np.cos(angles),
        np.sin(angles),
        angles*0
    ]).T
    points *= radius

    # 2) rotate circle
    points = rotate_from_axis_to_axis(
        points,
        [0, 0, 1],
        normal
    )

    # 3) add center
    points += center

    # 4) add noise
    points += np.random.normal(0, noise_sigma, points.shape)
    return points

