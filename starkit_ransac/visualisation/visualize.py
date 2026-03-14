import open3d as o3d
import numpy as np
from starkit_ransac.surfaces import (
        Circle2D,
        Line3D,
        Plane3D,
        Circle3D,
        Sphere,
        Ellipsoid3D,
        MobiusStrip,
        StepPlane
)

from starkit_ransac.visualisation import (
    generate_circle2D_mesh,
    generate_line3d_mesh,
    generate_plane_mesh,
    generate_circle_mesh,
    generate_sphere_mesh,
    generate_ellipsoid_mesh,
    generate_mobius_mesh,
    generate_stairs_mesh
)

TYPE_TO_GENERATOR = {
        Circle2D : generate_circle2D_mesh,
        Line3D : generate_line3d_mesh,
        Plane3D : generate_plane_mesh,
        Circle3D : generate_circle_mesh,
        Sphere : generate_sphere_mesh,
        Ellipsoid3D : generate_ellipsoid_mesh,
        StepPlane : generate_stairs_mesh,
        MobiusStrip : generate_mobius_mesh
}

def generate_mesh(
        surface,
        resolution=100,
        color=(0, 1, 0)
    ):
    surface_type = type(surface)
    func = TYPE_TO_GENERATOR[surface_type]
    return func(surface, resolution=resolution, color=color)

def setup_visualizer(winname='RASNAC'):
    o3d.visualization.gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer(winname, 1024, 768)
    color = np.full(4, 0.2)
    color[-1] = 1
    vis.set_background(color, None)
    vis.show_skybox(False)
    vis.line_width = 15
    vis.setup_camera(
        80,
        [0,0,0],
        [15, 0, 0],
        [0,0,1]
    )
    return vis

def draw_pretty(
        geom,
        line_width=7,
        point_size=2
    ):
    o3d.visualization.draw(
            geom,
            bg_color=(0.2, 0.2, 0.2, 1),
            show_skybox=False,
            line_width=line_width,
            point_size=point_size
    )
