import open3d as o3d
import numpy as np
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D

def generate_ellipsoid_mesh(
        ellipsoid:Ellipsoid3D,
        color=[0, 1, 0],
        resolution:int=50
    ):
    # x = a sin(theta) cos(phi)
    # y = b sin(theta) sin(phi)
    # z = c cos(theta)
    thetas = np.linspace(-np.pi, np.pi, resolution)
    phis = np.linspace(-np.pi, np.pi, resolution)

    radii = ellipsoid.model['radii']
    a, b, c = radii
    points = []
    lines = []
    i = 0
    for th in thetas:
        for phi in phis:
            cur_point = [
                a * np.sin(th) * np.cos(phi),
                b * np.sin(th) * np.sin(phi),
                c * np.cos(th)
            ]
            points.append(cur_point)
            lines.append([i, i+resolution])
            lines.append([i, i+1])
            i += 1

    # FIXME this is dirty, but whatever
    lines = lines[:-resolution*2]
    points = np.array(points)

    rotation = ellipsoid.model['axes'].T
    points = points @ rotation.T

    points += ellipsoid.model['center']

    mesh = o3d.geometry.LineSet()
    mesh.points = o3d.utility.Vector3dVector(points)
    mesh.lines = o3d.utility.Vector2iVector(lines)
    mesh.paint_uniform_color(color)
    return mesh

