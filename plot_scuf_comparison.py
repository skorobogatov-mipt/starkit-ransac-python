"""Plot benchmark comparison between scuf and starkit-ransac.

Usage:
    python scripts/plot_benchmark.py <benchmark.json> [--output <file.png>]
"""

import argparse
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_benchmarks(path):
    with open(path) as f:
        data = json.load(f)

    # group by library: {library_name: {n_iter: mean_time}}
    results = defaultdict(dict)

    for bench in data["benchmarks"]:
        name = bench["name"]
        n_iter = int(bench["params"]["n_iter"])
        mean = bench["stats"]["mean"]

        if "starkit_ransac" in name:
            results["starkit-ransac"][n_iter] = mean
        elif "scuf" in name:
            results["scuf"][n_iter] = mean

    return results


def plot(results, output):
    fig, ax = plt.subplots(figsize=(8, 5))

    for lib, timings in sorted(results.items()):
        iters = sorted(timings.keys())
        means = [timings[i] for i in iters]
        ax.plot(iters, means, marker="o", label=lib)

    ax.set_xlabel("RANSAC iterations")
    ax.set_ylabel("Time per run (s)")
    ax.set_title("RANSAC benchmark: scuf vs starkit-ransac")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_json", help="Path to pytest-benchmark JSON file")
    parser.add_argument("--output", "-o", help="Output image path (shows plot if omitted)")
    args = parser.parse_args()

    results = parse_benchmarks(args.benchmark_json)

    if not results:
        print("No benchmark entries found in the file.", file=sys.stderr)
        sys.exit(1)

    plot(results, args.output)


if __name__ == "__main__":
    main()
