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

def setup_visualizer():
    viz = o3d.visualization.Visualizer()
    viz.create_window()

    opt = viz.get_render_option()
    opt.background_color = np.array([0.2, 0.2, 0.2])
    opt.line_width = 10.
    opt.point_size = 5
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True
    return viz
