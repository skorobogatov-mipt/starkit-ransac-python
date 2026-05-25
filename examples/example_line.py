import open3d as o3d
import numpy as np
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.line3d import Line3D
from starkit_ransac.generators.line3d import generate_line3d
from starkit_ransac.visualisation.line3d import generate_line3d_mesh
from starkit_ransac.visualisation.visualize import draw_pretty, setup_visualizer, PCD_COLOR



def main():
    perfect_line = Line3D(direction=[0.5, 0.5, 0.5], point=[1, 0, -2.4])
    data = generate_line3d(perfect_line)
    ransac = RANSAC(data)
    model = ransac.fit(object_type=Line3D, iter_num=100, distance_threshold=0.1)

    avg = np.mean(data, axis=0)
    mesh = generate_line3d_mesh(model, length=7, midpoint=avg)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color(PCD_COLOR)

    draw_pretty(
        [mesh, pcd],
        filename='./figures/line3d.png'
    )


if __name__ == "__main__":
    main()
