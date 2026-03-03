import pytest

from starkit_ransac.ransac_3d import RANSAC3D
from starkit_ransac.surfaces.stairs import StepPlane
import numpy as np

# Parameters of stairs
N_POINTS = 350
N_STEPS = 6
STEP_HEIGHT = 0.2
STEP_WIDTH = 1.0
STAIR_SPAN = 0.4
ROTATION_DEG = 20
NOISE_SIGMA = 0.01
SEED = 42


@pytest.fixture
def stairs_data():
    # The generated cloud indicates stairs with noise
    np.random.seed(SEED)
    return StepPlane.generate_ladder_points(
        n_steps=N_STEPS,
        step_height=STEP_HEIGHT,
        step_width=STEP_WIDTH,
        stair_span=STAIR_SPAN,
        n_points=N_POINTS,
        noise_sigma=NOISE_SIGMA,
        rotation_deg=ROTATION_DEG,
    )


@pytest.fixture
def acceptable_rmse():
    return 0.08


def test_stairs(
        stairs_data,
        acceptable_rmse,
        ):
    # Fitting model stairs RANSAC
    runsuck = RANSAC3D()
    runsuck.add_points(stairs_data)

    np.random.seed(SEED)
    model = runsuck.fit(
            StepPlane,
            1200,
            0.06
    )
    result = model.get_model()
    print("RESULT : ", result)
    distances = model.calc_distances(stairs_data)
    rmse = np.sqrt(np.mean(distances ** 2))

    assert np.isfinite(result['step_width']) and result['step_width'] > 0
    assert np.isfinite(result['step_height']) and result['step_height'] > 0
    assert np.isfinite(result['rotation_deg'])
    assert rmse < acceptable_rmse
