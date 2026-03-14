import open3d as o3d
import numpy as np
from starkit_ransac.surfaces.plane import Plane3D
from starkit_ransac.generators.plane import generate_plane
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.visualisation import generate_mesh, setup_visualizer


def main():
    perfect_plane = Plane3D(
        a=np.random.random(),
        b=np.random.random(),
        c=np.random.random(),
        d=np.random.random()
    )
    data = generate_plane(perfect_plane)
    
    ransac = RANSAC(data)
    fit_plane = ransac.fit(Plane3D, 1000, 0.05)

    mesh = generate_mesh(fit_plane)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.9]*3)

    viz = setup_visualizer()
    viz.add_geometry(pcd)
    viz.add_geometry(mesh)

    viz.run()


if __name__ == "__main__":
    main()
