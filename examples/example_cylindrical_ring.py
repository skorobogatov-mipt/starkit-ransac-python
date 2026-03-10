import open3d as o3d
import matplotlib.pyplot as plt

from starkit_ransac.generators.generators import generate_cylindrical_ring
from starkit_ransac.visualisation.cylindircal_ring import visualize_ring
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.cylindrical_ring import CylindricalRing

def main():
    data = generate_cylindrical_ring(
        inner_r=0.2,
        outer_r=1.,
        height=0.5,
        center=(0.42, -0.3, 7),
        n_points=1000
    )
    ransac = RANSAC()
    ransac.add_points(data)
    model = ransac.fit(
        CylindricalRing,
        1000,
        0.1
    )
    visualize_ring(data, model)

if __name__ == "__main__":
    main()
