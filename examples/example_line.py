import open3d as o3d
from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.surfaces.line3d import Line3D
from starkit_ransac.generators.line3d import generate_line3d
from starkit_ransac.visualisation.line3d import generate_line3d_mesh

def main():
    perfect_line = Line3D(
        direction=[0.5, 0.5, 0.5],
        point=[1, 0, -2.4]
    )
    data = generate_line3d(perfect_line)
    ransac = RANSAC3D(data)
    model = ransac.fit(
            object_type=Line3D,
            iter_num=100,
            distance_threshold=0.1
    )
    
    mesh = generate_line3d_mesh(model, length=10)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == "__main__":
    main()
