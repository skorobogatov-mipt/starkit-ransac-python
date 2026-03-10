import numpy as np
import matplotlib.pyplot as plt
from starkit_ransac.surfaces.ellipse2d import Ellipse2D
from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.generators.ellipse2d import generate_ellipse2d

def main():
    perfect_ellipse = Ellipse2D(
            rotation=np.random.random()*np.pi*2,
            radius_1=np.random.random(1)*5,
            radius_2=np.random.random(1)*10,
            center=np.random.random(2)*5
    )
    data = generate_ellipse2d(perfect_ellipse)

    ransac = RANSAC3D(data)
    fit_ellipse = ransac.fit(
            Ellipse2D,
            500,
            0.05
    )       
    
    fit_data = generate_ellipse2d(fit_ellipse, noise_sigma=0)
    plt.scatter(data[:, 0], data[:, 1], c=(0.5, 0.5, 0.5))
    plt.plot(fit_data[:, 0], fit_data[:, 1], c=(0, 1, 0))
    plt.show()



if __name__ == "__main__":
    main()

