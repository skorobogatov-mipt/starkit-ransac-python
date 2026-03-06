import open3d as o3d
import numpy as np

from starkit_ransac.generators.ellipsoid import generate_ellipsoid
from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D
from starkit_ransac.visualisation.ellipsoid import generate_ellipsoid_mesh

def main():
    # np.random.seed(42)
    v1 = np.random.random(3) - 0.5
    v2 = np.random.random(3) - 0.5
    v3 = np.cross(v1, v2)
    v1 = np.cross(v2, v3)
    axes = np.asarray([v1, v2, v3])
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

    perfect_ellipsoid = Ellipsoid3D(
        axes=axes,
        radii=[5, 2, 3],
        center=np.random.random(3)*5
    )
    data = generate_ellipsoid(
            perfect_ellipsoid,
            n_points=5000,
    )
    ransac = RANSAC3D(data)
    model:Ellipsoid3D = ransac.fit(
        Ellipsoid3D,
        10,
        0.1
    )

    mesh = generate_ellipsoid_mesh(model, resolution=25)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == "__main__":
    main()
