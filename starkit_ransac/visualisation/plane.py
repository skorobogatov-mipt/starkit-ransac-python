import open3d as o3d
import numpy as np
from starkit_ransac.surfaces.plane import Plane3D
from starkit_ransac.utils import normalize

def generate_plane_mesh(
        plane:Plane3D,
        color=[0, 1, 0],
        approx_center=[0, 0, 0],
        size=10
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

    p1 = center + v1 * size
    p2 = center + v2 * size
    p3 = center - v1 * size
    p4 = center - v2 * size
    points = np.array([p1,p2,p3,p4])
    triangles = [[0,1,2],[0,2,3]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    return mesh



