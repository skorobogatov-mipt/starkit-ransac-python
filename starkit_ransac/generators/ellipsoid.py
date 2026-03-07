import pdb
import numpy as np
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D

def generate_ellipsoid(
        ellipsoid:Ellipsoid3D,
        noise_sigma:float=0.05,
        n_points=1000
    ):

    # FIXME this method generates point clouds with a bit of inconsistent
    # density
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

def generate_ellipsoid_poly(
        polynomial,
        resolution=100,
        noise_sigma=0.05,
        x_range=(-10, 10),
        y_range=(-10, 10),
        z_range=(-10, 10),
        precision=0.05
    ):
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    zs = np.linspace(*z_range, resolution)
    X,Y,Z = np.meshgrid(xs, ys, zs)
    
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    distances = np.dot(
        polynomial,
        np.array([
            x**2, 
            y**2, 
            z**2, 
            x*y,
            x*z, 
            y*z, 
            x, 
            y, 
            z
        ])
    ) - 1
    selected_xs = x[np.abs(distances) < precision]
    selected_ys = y[np.abs(distances) < precision]
    selected_zs = z[np.abs(distances) < precision]
    if len(selected_xs) == 0:
        return []
    points = np.column_stack((selected_xs, selected_ys, selected_zs))
    return points


