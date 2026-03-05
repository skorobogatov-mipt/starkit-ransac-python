import numpy as np
import open3d as o3d
from starkit_ransac.surfaces import circle2d
from starkit_ransac.surfaces.circle2d import Circle2D
from starkit_ransac.generators.circle2d import generate_circle2D

def generate_circle2D_mesh(
        circle:Circle2D,
        color=np.array([0, 1, 0]),
        resolution=100
    ):
    points = generate_circle2D(
            circle, 
            noise_sigma=0,
            n_points=resolution
    )
    z = np.zeros(resolution).reshape((resolution, 1))
    points = np.hstack((points, z))
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(points)
    mesh.paint_uniform_color(color)

    return mesh

