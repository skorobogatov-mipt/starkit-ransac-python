import numpy as np
import open3d as o3d
from starkit_ransac.generators.circle import generate_circle
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.circle import Circle3D
from starkit_ransac.visualisation import generate_mesh, setup_visualizer

def main():
    perfect_model = Circle3D(
      np.random.random(3) * 20,
      5.5,
      np.random.random(3)
    )
    data = generate_circle(
            perfect_model,
            1.5,
            6000,
    )

    ransac = RANSAC()
    ransac.add_points(data)

    model = ransac.fit(
        Circle3D,
        6000,
        0.05
    )


    viz = setup_visualizer()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.9]*3)

    mesh = generate_mesh(model, color=[0, 1, 0])

    viz.add_geometry(pcd)
    viz.add_geometry(mesh)
    viz.run()

if __name__ == "__main__":
    main()
