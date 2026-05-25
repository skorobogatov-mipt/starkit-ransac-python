import numpy as np
import pytest
from filelock import FileLock
import json
import os

SEED = 42

np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

N_ITER_BENCHMARK = 500
BENCHMARK_THRESH = 0.1

class AbstractGenerator:
    @pytest.fixture(scope="class")
    def data_points(self, perfect_model, noise_sigma, n_points):
        return generate_line3d(
            perfect_model, noise_sigma=noise_sigma, n_points=n_points
        )

class AbstractTestPrecision:
    n_iter_list = [500, 1000, 2000, 3000]
    N_ITER_AVERAGE = 10
    THRESHOLD = 0.1

    @pytest.fixture(scope="class", params=n_iter_list)
    def n_iter(self, request):
        return request.param
    
    RESULT_PATH = 'woow_precision_comparison.json'
    LOCK_PATH = RESULT_PATH + '.lock'

    @staticmethod
    def generate_result_dict(
            n_iter:int,
            starkit_ransac_rmse:float,
            pyransac_rmse:float,
            n_data:int,
            shape_name:str,
            scuf_rmse:float | None = None,
            ):
        result = {
            'n_iter' : n_iter,
            'starkit_ransac RMSE' : starkit_ransac_rmse,
            'pyransac RMSE' : pyransac_rmse,
            'n_data' : n_data,
            'shape_name' : shape_name
        }
        if scuf_rmse is not None:
            result['scuf RMSE'] = scuf_rmse
        return result
    
    buffer = []
    buffer_period = 100
    # Stagger flushes across xdist workers so they don't all hit the lock together.
    _worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'gw0')
    n_buffer_writes = int(''.join(c for c in _worker_id if c.isdigit()) or 0) % buffer_period
    @staticmethod
    def _append_result(result):
        AbstractTestPrecision.n_buffer_writes += 1
        if AbstractTestPrecision.n_buffer_writes % AbstractTestPrecision.buffer_period == 0:
            with FileLock(AbstractTestPrecision.LOCK_PATH):
                path = AbstractTestPrecision.RESULT_PATH
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                else:
                    data = []
                data.extend(AbstractTestPrecision.buffer)
                data.append(result)
                AbstractTestPrecision.buffer = []
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
        else:
            AbstractTestPrecision.buffer.append(result)

    @staticmethod
    def _flush_buffer():
        if not AbstractTestPrecision.buffer:
            return
        with FileLock(AbstractTestPrecision.LOCK_PATH):
            path = AbstractTestPrecision.RESULT_PATH
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            data.extend(AbstractTestPrecision.buffer)
            AbstractTestPrecision.buffer = []
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)


def pytest_sessionfinish(session, exitstatus):
    AbstractTestPrecision._flush_buffer()
