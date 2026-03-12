import open3d as o3d
import numpy as np
from starkit_ransac.surfaces.circle import Circle3D
from starkit_ransac.generators.circle import generate_circle

def generate_circle_mesh(
        circle:Circle3D,
        color=np.array([0, 1, 0]),
        resolution=100
    ):
    points = generate_circle(
            circle, 
            noise_sigma=0.,
            n_points=resolution
    )
    lines = [[i, (i + 1) % resolution] for i in range(resolution)]

    mesh = o3d.geometry.LineSet()
    mesh.points = o3d.utility.Vector3dVector(points)
    mesh.lines = o3d.utility.Vector2iVector(lines)
    mesh.paint_uniform_color(color)
    return mesh

