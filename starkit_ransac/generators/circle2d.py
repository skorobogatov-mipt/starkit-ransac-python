import numpy as np
from starkit_ransac.surfaces.circle2d import Circle2D

def generate_circle2D(
        circle:Circle2D,
        noise_sigma:float=0.05,
        n_points=1000
    ):
    center = circle.model['center']
    radius = circle.model['radius']

    angles = np.linspace(-np.pi, np.pi, n_points)
    points = np.vstack([
        np.cos(angles),
        np.sin(angles)
    ]).T
    points *= radius
    points += center

    noise = np.random.normal(0, noise_sigma, points.shape)

    return points + noise

