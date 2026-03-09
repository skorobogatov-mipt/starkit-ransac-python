import numpy as np
from starkit_ransac.surfaces.sphere import Sphere
from starkit_ransac.utils import normalize

def generate_sphere(
        sphere:Sphere,
        noise_sigma=0.05,
        n_points=1000
    ):

    points = np.random.random((n_points, 3)) - 0.5
    points = normalize(points, -1)

    points *= sphere.radius
    
    points += sphere.center

    points += np.random.normal(0, noise_sigma, points.shape)

    return points
