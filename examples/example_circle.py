import numpy as np
import open3d as o3d
from starkit_ransac.generators.circle import generate_circle
from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.surfaces.circle import Circle
from starkit_ransac.visualisation.circle import generate_circle_mesh

def main():
    np.random.seed(42)
    # perfect_model = Circle(
    #     [0.1315345, 0.2264877, 0.54367253],
    #     3.2,
    #     [0.19601301, 0.69700544, 0.68975525]
    #
    # )
    perfect_model = Circle(
      [ 0.23687941, -0.395462,    0.96620461],
      2.7896038304338235,
      [-0.163813,   -0.58250497, -0.79614902],
    )
    data = generate_circle(perfect_model)

    ransac = RANSAC3D()
    ransac.add_points(data)

    model = ransac.fit(
            Circle,
            1500,
            0.05
    )

    print(perfect_model)
    print(model)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.paint_uniform_color([0.6]*3)
    mesh = generate_circle_mesh(model, [1, 0, 0])
    o3d.visualization.draw_geometries([pcd,mesh])

if __name__ == "__main__":
    main()
