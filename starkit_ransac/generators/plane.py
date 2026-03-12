from starkit_ransac.surfaces.plane import Plane3D
import numpy as np
from starkit_ransac.utils import normalize

def generate_plane(
        plane:Plane3D,
        noise_sigma=0.05,
        n_points=1000, 
        plane_size=10,
        approx_center=[0,0,0]
    ):
    a = plane.a
    b = plane.b
    c = plane.c
    d = plane.d

    normal = np.array([a,b,c])
    normal /= np.linalg.norm(normal)
    
    distance = plane.calc_distance_one_point(approx_center)
    
    direction = np.sign(np.dot([a,b,c], approx_center) + d)
    center = approx_center - direction * distance * normal

    # get some vector not parallel to a normal
    q1 = np.copy(normal)
    q1[:] = q1[::-1] # just reverse the normal

    # get two orthogonal vectors that lie in the plane
    v1 = np.cross(normal, q1)
    v2 = np.cross(normal, v1)

    v1 = normalize(v1)
    v2 = normalize(v2)

    point_combos = np.random.random((n_points, 2)) - 0.5
    point_combos = normalize(point_combos, axis=-1)
    point_combos *= np.random.random((n_points, 1)) * plane_size

    points = np.outer(point_combos[:, 0], v1) + np.outer(point_combos[:, 1], v2)

    points += np.random.normal(0, noise_sigma, points.shape)

    return points
