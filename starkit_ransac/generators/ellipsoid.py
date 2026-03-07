import pdb
import numpy as np
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D

def generate_ellipsoid(
        ellipsoid:Ellipsoid3D,
        noise_sigma:float=0.05,
        n_points=1000
    ):

    # an ellipsoid has three rotation angles
    axes = ellipsoid.model['axes']
    rotation_matrix = axes.T

    radii = ellipsoid.model['radii']
    center = ellipsoid.model['center']

    # generate n random unit vectors
    points = np.random.random((n_points, 3)) - 0.5
    points = points / np.linalg.norm(points, axis=-1, keepdims=True)

    # stretch these unit vectors by radii
    points[..., 0] *= radii[0]
    points[..., 1] *= radii[1]
    points[..., 2] *= radii[2]

    # rotate points
    points = points @ rotation_matrix.T

    points += center

    points += np.random.normal(0, noise_sigma, points.shape)

    return points

# def generate_ellipsoid_poly(
#         ellipsoid:Ellipsoid3D,
#         noise_sigma:float=0.05,
#         n_points=1000
#     ):
#     poly = ellipsoid[]
