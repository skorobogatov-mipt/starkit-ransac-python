import numpy as np
import matplotlib.pyplot as plt
from starkit_ransac.generators.circle2d import generate_circle2D
from starkit_ransac.surfaces.circle2d import Circle2D
from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.visualisation.circle2d import generate_circle2D_mesh

def main():
    ideal = Circle2D(
        center=(0.5, 0.1),
        radius=1.2
    )
    data = generate_circle2D(
            ideal,
            noise_sigma=0.1
    )

    ransac = RANSAC()
    ransac.add_points(data)
    model = ransac.fit(
            Circle2D,
            iter_num=1000,
            distance_threshold=0.05
    )

    fit_data = generate_circle2D(model, noise_sigma=0)

    plt.scatter(data[:, 0], data[:, 1], c=(0.5, 0.5, 0.5))
    plt.plot(fit_data[:, 0], fit_data[:, 1], c=(0, 1, 0))
    plt.gca().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()
