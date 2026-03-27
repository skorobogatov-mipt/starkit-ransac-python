import numpy as np
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import matplotlib


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("logfile")
    return parser.parse_args()


def is_benchmark_starkit(benchmark):
    return "test_benchmark_starkit_ransac" in benchmark["name"]


def is_benchmark_pyransac(benchmark):
    return "test_benchmark_pyransac" in benchmark["name"]


def is_benchmark_scuf(benchmark):
    return "test_benchmark_scuf" in benchmark["name"]

def which_library(benchmark, libraries):
    for lib in libraries:
        test_name = "test_benchmark_" + lib
        if test_name in benchmark["name"]:
            return lib
    return None

def which_test(benchmark, targets):
    for target in targets:
        if "test_"+target in benchmark["fullname"]:
            return target
    return None


# COLOR = "white"
# matplotlib.rcParams["text.color"] = COLOR
# matplotlib.rcParams["axes.labelcolor"] = COLOR
# matplotlib.rcParams["xtick.color"] = COLOR
# matplotlib.rcParams["ytick.color"] = COLOR

def plot_comparison(lib1, lib2, avgs, w=0.4):
    lib2_shapes = list(avgs[lib2].keys())
    n_shapes_lib2 = len(lib2_shapes)
    lib2_bar = np.arange(n_shapes_lib2) + w
    lib1_bar = lib2_bar - w
    
    lib1_avgs = []
    for shape in lib2_shapes:
        lib1_avgs.append(avgs[lib1][shape])

    plt.bar(lib1_bar, lib1_avgs, w, label=lib1)
    plt.bar(lib2_bar, avgs[lib2].values(), w, label=lib2)
    plt.xticks(lib2_bar - w / 2, lib2_shapes, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Time per RANSAC run, s", fontsize=32)
    plt.xlabel("Shapes", fontsize=32)
    plt.legend(fontsize=32)
    plt.show()

def main():
    file = parse_args().logfile
    with open(file, "r") as inp:
        benchmark_data = json.load(inp)

    target_tests = [
        "line",
        "plane",
        "circle",
        "sphere",
        "ellipsoid",
    ]
    target_libraries = [
        "starkit_ransac",
        "pyransac",
        "scuf"

    ]

    n_iter = {}
    total_time = {}

    for lib in target_libraries:
        total_time[lib] = {}
        n_iter[lib] = {}
        for test in target_tests:

            total_time[lib][test] = 0
            n_iter[lib][test] = 0

    for benchmark in benchmark_data["benchmarks"]:
        lib = which_library(benchmark, target_libraries)
        test = which_test(benchmark, target_tests)
        time = benchmark['stats']['mean']
        n_iter[lib][test] += 1
        total_time[lib][test] += time

    
    avgs = {}
    collected_shapes = []

    for lib in target_libraries:
        avgs[lib] = {}
        for test in target_tests:
            if n_iter[lib][test] == 0:
                continue

            if lib == "starkit_ransac":
                if total_time[lib][test] != 0:
                    collected_shapes.append(test)
            
            avgs[lib][test] = total_time[lib][test] / n_iter[lib][test]

    n_spaces = 15
    print(' '*n_spaces, end='')
    for shape in target_tests:
        print(shape, end='    ')
    print('\n', end='')

    for lib in target_libraries:
        print(lib, end='')
        print(' '*(n_spaces - len(lib)), end='')
        for shape in target_tests:
            if shape in avgs[lib].keys():
                to_print = format(avgs[lib][shape], '.4f')
                print(to_print, end=' '*(4 + len(shape) - len(to_print)))
            else:
                print('-'*len(shape), end='    ')
            # print(' ', end='')

        print('')

    plot_comparison('starkit_ransac', 'pyransac', avgs)
    plot_comparison('starkit_ransac', 'scuf', avgs)



if __name__ == "__main__":
    main()
