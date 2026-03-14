import numpy as np
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import matplotlib

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('logfile')
    return parser.parse_args()

def is_benchmark_starkit(benchmark):
    return "test_benchmark_starkit_ransac" in benchmark['name']

def is_benchmark_pyransac(benchmark):
    return "test_benchmark_pyransac" in benchmark['name']

def which_target(benchmark, targets):
    for target in targets:
        if target in benchmark['fullname']:
            return target
    return None


COLOR = 'white'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR

def main():
    file = parse_args().logfile
    with open(file, 'r') as inp:
        benchmark_data = json.load(inp)

    target_tests = [
            'test_circle',
            'test_line',
            'test_sphere',
            'test_plane'
    ]
    total_time_starkit = {}
    total_time_pyransac = {}
    n_iter_stransac = {}
    n_iter_pyransac = {}
    for test in target_tests:
        total_time_starkit[test] = 0
        n_iter_stransac[test] = 0

        total_time_pyransac[test] = 0 
        n_iter_pyransac[test] = 0

    for benchmark in benchmark_data['benchmarks']:
        test = which_target(benchmark, target_tests)
        if test is None:
            continue
        if is_benchmark_starkit(benchmark):
            n_iter_stransac[test] += 1
            total_time_starkit[test] += benchmark['stats']['mean']
        elif is_benchmark_pyransac(benchmark):
            n_iter_pyransac[test] += 1
            total_time_pyransac[test] += benchmark['stats']['mean']

    stransac_avgs = []
    pyransac_avgs = []
    collected_shapes = []
    for test in target_tests:
        if n_iter_stransac[test] == 0:
            continue

        shape = test.removeprefix('test_')
        collected_shapes.append(shape)
        stransac_avgs.append(
            total_time_starkit[test]/n_iter_stransac[test]
        )

        pyransac_avgs.append(
            total_time_pyransac[test]/n_iter_pyransac[test]
        )

    w = 0.4
    stransac_bar = np.arange(len(collected_shapes))
    pyransac_bar = stransac_bar + w
    plt.bar(
            stransac_bar, 
            stransac_avgs, 
            w,
            label='starkit_ransac',
            color=(0,1,0)
    )
    plt.bar(
            pyransac_bar, 
            pyransac_avgs, 
            w,
            label='pyransac-3d',
            color=(1,0,0)
    )
    plt.xticks(stransac_bar + w/2, collected_shapes, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel('Time per iteration, s', fontsize=32)
    plt.xlabel('Shapes', fontsize=32)

    plt.gca().set_facecolor((0.2, 0.2, 0.2))
    plt.gcf().set_facecolor((0.2, 0.2, 0.2))
    plt.legend(facecolor=(0.4, 0.4, 0.4), fontsize=32)
    plt.show()


if __name__ == "__main__":
    main()
