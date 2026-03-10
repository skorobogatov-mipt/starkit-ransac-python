import os
import numpy as np
import open3d as o3d
import math

from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.stairs import StepPlane
from starkit_ransac.generators.generators import generate_stairs
from starkit_ransac.visualisation.stairs import generate_stairs_mesh, visualize_stairs

# Parameters of stairs
N_POINTS = 650
N_STEPS = 3
STEP_HEIGHT = 0.5
STEP_WIDTH = 0.6
STAIR_SPAN = 0.4
ROTATION_DEG = 20
NOISE_SIGMA = 0
SEED = 42

def main():
    data = generate_stairs(
        n_steps=5,
        step_height=0.3,
        step_width=0.5,
        stair_span=2,
        n_points=1000,
        rotation_deg=30,
        noise_sigma=0.05
    )
    ransac = RANSAC()
    ransac.add_points(data)
    model = ransac.fit(
            StepPlane,
            1200,
            0.06
    )
    visualize_stairs(data, model)


if __name__ == "__main__":
    main()
