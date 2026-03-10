import open3d as o3d
import numpy as np
from starkit_ransac.generators.circle2d import generate_circle2D
from starkit_ransac.surfaces.circle2d import Circle2D
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.visualisation.circle2d import generate_circle2D_mesh

def main():
    ideal = Circle2D(
        center=(0.5, 0.1),
        radius=1.7
    )
    data = generate_circle2D(
            ideal,
            noise_sigma=0.1
    )

    ransac = RANSAC()
    ransac.add_points(data)
    model = ransac.fit(
            Circle2D,
            iter_num=500,
            distance_threshold=0.05
    )

    mesh = generate_circle2D_mesh(model)

    z = np.zeros(len(data)).reshape((len(data), 1))
    data = np.hstack((data,z))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == "__main__":
    main()
