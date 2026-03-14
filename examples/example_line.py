import open3d as o3d
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.line3d import Line3D
from starkit_ransac.generators.line3d import generate_line3d
from starkit_ransac.visualisation.line3d import generate_line3d_mesh
from starkit_ransac.visualisation.visualize import setup_visualizer

def main():
    perfect_line = Line3D(
        direction=[0.5, 0.5, 0.5],
        point=[1, 0, -2.4]
    )
    data = generate_line3d(perfect_line)
    ransac = RANSAC(data)
    model = ransac.fit(
            object_type=Line3D,
            iter_num=100,
            distance_threshold=0.1
    )
    

    mesh = generate_line3d_mesh(model, length=20)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.9, 0.9, 0.9])

    viz = setup_visualizer()

    viz.add_geometry(pcd)
    viz.add_geometry(mesh)
    viz.run()


if __name__ == "__main__":
    main()
