import open3d as o3d
from starkit_ransac.surfaces import line3d
from starkit_ransac.surfaces.line3d import Line3D


def generate_line3d_mesh(
        line:Line3D,
        color=[0, 1, 0],
        length=5,
        midpoint=None,
        resolution=None
    ):
    if midpoint is None:
        midpoint = line.point
    p0 = midpoint - line.direction * length/2
    p1 = midpoint + line.direction * length/2

    mesh = o3d.geometry.LineSet()
    mesh.points = o3d.utility.Vector3dVector([p0, p1])
    mesh.lines = o3d.utility.Vector2iVector([[0, 1]])
    mesh.paint_uniform_color(color)
    return mesh
