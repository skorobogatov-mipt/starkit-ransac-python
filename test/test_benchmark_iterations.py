import pytest
import numpy as np

from starkit_ransac.ransac_3d import RANSAC
from starkit_ransac.surfaces.ellipsoid import Ellipsoid3D
from starkit_ransac.generators.ellipsoid import generate_ellipsoid
from starkit_ransac.utils import normalize

import scuf

from conftest import BENCHMARK_THRESH, RNG


N_ITERATIONS_LIST = [100, 500, 1000, 2500, 5000, 10000]


@pytest.fixture(scope="module")
def ellipsoid_data():
    """Generate a single ellipsoid dataset shared across all benchmarks."""
    v1 = RNG.random(3) - 0.5
    v2 = RNG.random(3) - 0.5
    v3 = np.cross(v1, v2)
    v1 = np.cross(v2, v3)
    v2 = np.cross(v3, v1)
    axes = np.array([v1, v2, v3])
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

    model = Ellipsoid3D(axes=axes, radii=[3.6, 3.0, 5.0], center=RNG.random(3) * 10)
    data = generate_ellipsoid(model, n_points=5000, noise_sigma=0.05)
    return data


@pytest.mark.parametrize("n_iter", N_ITERATIONS_LIST)
def test_benchmark_starkit_ransac_iterations(ellipsoid_data, n_iter, benchmark):
    ransac = RANSAC(ellipsoid_data)
    benchmark(ransac.fit, Ellipsoid3D, n_iter, BENCHMARK_THRESH)


@pytest.mark.parametrize("n_iter", N_ITERATIONS_LIST)
def test_benchmark_scuf_iterations(ellipsoid_data, n_iter, benchmark):
    IS = scuf.ransac.RANSAC(figure="ellipsoid")
    benchmark(
        IS.fit, ellipsoid_data, iterations=n_iter, threshold=BENCHMARK_THRESH
    )
