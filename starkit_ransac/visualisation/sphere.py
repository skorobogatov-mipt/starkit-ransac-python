import numpy as np
from starkit_ransac.surfaces.sphere import Sphere
import open3d as o3d

def generate_sphere_mesh(
        sphere:Sphere,
        color=[0, 1, 0],
        resolution:int=50
    ):
    R = sphere.radius
    hs = np.linspace(-R, R, resolution)
    thetas = np.linspace(-np.pi, np.pi, resolution)
    
    r_aux = np.sqrt(R**2 - hs**2)

    points = []
    lines = []
    i = 0
    for h, r in zip(hs, r_aux):
        for th in thetas:
            points.append([
                r * np.cos(th),
                r * np.sin(th),
                h
            ])
            lines.append([i, i+resolution])
            lines.append([i, i+1])
            i += 1
    points += sphere.center

    lines = lines[:-resolution*2]

    mesh = o3d.geometry.LineSet()
    mesh.points = o3d.utility.Vector3dVector(points)
    mesh.lines = o3d.utility.Vector2iVector(lines)
    mesh.paint_uniform_color(color)
    return mesh
    
