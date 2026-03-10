import numpy as np
from starkit_ransac.surfaces.ellipse2d import Ellipse2D

def generate_ellipse2d(
        ellipse:Ellipse2D,
        noise_sigma=0.05,
        n_poins=1000
    ):
    thetas = np.linspace(-np.pi, np.pi, n_poins)
    a = ellipse.major_radius
    b = ellipse.minor_radius
    points = np.array([
        a * np.cos(thetas),
        b * np.sin(thetas)
    ]).T
    rot = ellipse.rotation
    rotation_matr = np.array([
        [np.cos(rot), -np.sin(rot)],
        [np.sin(rot), np.cos(rot)]
    ])

    points = points @ rotation_matr.T
    points += ellipse.center
    points += np.random.normal(0, noise_sigma, points.shape)
    return points

