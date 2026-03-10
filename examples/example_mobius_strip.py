import open3d as o3d
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.Mobius_strip import Mobius_strip
from starkit_ransac.generators.Mobius_strip import generate_mobius
from starkit_ransac.visualisation.Mobius_strip import generate_mobius_mesh

def main():
    sample_mobius = Mobius_strip(
        center=[0, 0, 0],
        radius=5,
        normal=[1, 0, 0],
        orientation=1,
        start_vector=[0, 0, 1],
        width=2
    )
    data = generate_mobius(
        sample_mobius,
        n_points=1000,
        noise_sigma=0
    )
    ransac = RANSAC()
    ransac.add_points(data)
    fitted = ransac.fit(
        Mobius_strip,
        50,
        0.01
    )
    mesh = generate_mobius_mesh(fitted)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    o3d.visualization.draw_geometries(
            [pcd, mesh]
    )

if __name__ == "__main__":
    main()
