import open3d as o3d
import numpy as np

from starkit_ransac.generators import ellipsoid
from starkit_ransac.generators.ellipsoid import generate_ellipsoid, generate_ellipsoid_poly
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D
from starkit_ransac.visualisation.visualize import generate_mesh, setup_visualizer

def main():
    np.random.seed(42)
    v1 = np.random.random(3) - 0.5
    v2 = np.random.random(3) - 0.5
    v3 = np.cross(v1, v2)
    v1 = np.cross(v2, v3)
    axes = np.asarray([v1, v2, v3])
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

    perfect_ellipsoid = Ellipsoid3D(
        axes=axes,
        radii=[3.6,3,5],
        center=np.random.random(3)
    )
    data = generate_ellipsoid(
            perfect_ellipsoid,
            n_points=5000,
            noise_sigma=0.05
    )
    ransac = RANSAC(data)
    model:Ellipsoid3D = ransac.fit(
        Ellipsoid3D,
        5000,
        0.05
    )


    fitted_mesh = generate_mesh(model, 20)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.9, 0.9, 0.9])

    viz = setup_visualizer()

    viz.add_geometry(pcd)
    viz.add_geometry(fitted_mesh)
    viz.run()


if __name__ == "__main__":
    main()
