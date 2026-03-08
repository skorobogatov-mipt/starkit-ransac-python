import pdb
import numpy as np
from starkit_ransac.surfaces.line3d import Line3D
def generate_line3d(
        line:Line3D,
        noise_sigma:float=0.05,
        n_points=1000,
        distance_along_line=5
    ):
    direction = line.direction
    point = line.point
    t = np.linspace(-distance_along_line/2, distance_along_line/2, n_points)
    points = np.outer(t, direction) + point

    return points + np.random.normal(0, noise_sigma, points.shape)

