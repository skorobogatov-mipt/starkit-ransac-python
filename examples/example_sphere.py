import open3d as o3d
from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.surfaces.sphere import Sphere
from starkit_ransac.generators.sphere import generate_sphere
from starkit_ransac.visualisation.sphere import generate_sphere_mesh

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
    ransac = RANSAC3D(data)
    model = ransac.fit(Sphere, 10, 0.05)

    print(perfect_sphere.radius)
    print(model.center)

    mesh = generate_sphere_mesh(model)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([pcd, mesh])

if __name__ == "__main__":
    main()
