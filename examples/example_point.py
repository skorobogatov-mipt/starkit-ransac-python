import numpy as np
from starkit_ransac.surfaces.point import Point3D
from starkit_ransac.ransac_3d import RANSAC

SEED = 42
np.random.seed(SEED)

def generate_data(real_point, max_noise_abs=0.5):
    assert max_noise_abs >= 0
    noise = np.random.random((100, 3)) * 2 - 1
    noise = noise * max_noise_abs + real_point
    real_and_noise = np.concat(([real_point], noise), axis=0)
    return real_and_noise

def main():
    # generate synthetic data
    data = generate_data(np.zeros(3))

    ransac = RANSAC()
    ransac.add_points(data)
    fitted_model = ransac.fit(Point3D, 1000, 0.5)
    print(fitted_model)

if __name__ == "__main__":
    main()
