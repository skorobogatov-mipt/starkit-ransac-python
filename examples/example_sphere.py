import open3d as o3d
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.sphere import Sphere
from starkit_ransac.generators.sphere import generate_sphere
from starkit_ransac.visualisation.sphere import generate_sphere_mesh
from starkit_ransac.visualisation.visualize import generate_mesh, setup_visualizer

def main():
    perfect_sphere = Sphere(
        [1, 2, 3],
        6.9
    )
    data = generate_sphere(
            perfect_sphere,
            noise_sigma=0.1,
            n_points=1000
    )
    ransac = RANSAC(data)
    model = ransac.fit(Sphere, 10, 0.05)

    print(perfect_sphere.radius)
    print(model.center)

    mesh = generate_mesh(model, resolution=15)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.9]*3)

    viz = setup_visualizer()

    viz.add_geometry(pcd)
    viz.add_geometry(mesh)
    viz.run()

if __name__ == "__main__":
    main()
