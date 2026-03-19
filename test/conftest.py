import numpy as np
import pytest

SEED = 42

np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

N_ITER_BENCHMARK = 500
BENCHMARK_THRESH = 0.1
