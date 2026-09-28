#!/usr/bin/env python3
"""Summarize paired CUDA event measurements without discarding any samples."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(samples, values):
    median = statistics.median(samples)
    mean = statistics.mean(samples)
    return {
        "samples": len(samples),
        "median_us": median / 1000,
        "p10_us": percentile(samples, 0.1) / 1000,
        "p90_us": percentile(samples, 0.9) / 1000,
        "cv": statistics.pstdev(samples) / mean,
        "billion_values_per_second": values / median,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    samples = defaultdict(list)
    pairs = defaultdict(dict)
    sessions = defaultdict(list)
    metadata = []
    for path in args.logs:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["type"] == "machine":
                metadata.append(record)
                continue
            if record["type"] != "sample":
                continue
            workload = f"n{record['points']}_d{record['dimensions']}"
            route = record["route"]
            key = (workload, route)
            duration = record["event_ns"]
            samples[key].append(duration)
            sessions[(workload, route, record["session"])].append(duration)
            pairs[(workload, record["session"], record["repetition"])][route] = duration
    report = {
        "format": "sofl-public-curanddx-comparison-v1",
        "machine": metadata[0],
        "routes": {},
        "comparisons": {},
    }
    for (workload, route), durations in sorted(samples.items()):
        points = int(workload.split("_")[0][1:])
        dimensions = int(workload.split("_")[1][1:])
        data = summary(durations, points * dimensions)
        session_medians = [
            statistics.median(values)
            for (group, name, _), values in sessions.items()
            if group == workload and name == route
        ]
        data["session_median_cv"] = (
            statistics.pstdev(session_medians) /
            statistics.mean(session_medians)
        )
        report["routes"].setdefault(workload, {})[route] = data
    for workload, routes in report["routes"].items():
        if set(routes) != {
            "sofl", "curanddx_fresh", "curanddx_prepared", "curanddx_setup"
        }:
            raise SystemExit(f"missing routes for {workload}")
        result = {}
        for route in ("curanddx_fresh", "curanddx_prepared"):
            ratio = [
                measurements[route] / measurements["sofl"]
                for (group, _, _), measurements in pairs.items()
                if group == workload
            ]
            if len(ratio) != routes["sofl"]["samples"]:
                raise SystemExit(f"incomplete pairs for {workload}/{route}")
            result[f"sofl_over_{route}_median_speedup"] = (
                routes[route]["median_us"] / routes["sofl"]["median_us"]
            )
            result[f"sofl_over_{route}_paired_p10_speedup"] = (
                percentile(ratio, 0.1)
            )
        result["curanddx_setup_median_us"] = routes[
            "curanddx_setup"
        ]["median_us"]
        report["comparisons"][workload] = result
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for workload, comparison in sorted(report["comparisons"].items()):
        print(f"\n{workload}")
        for route in ("sofl", "curanddx_fresh", "curanddx_prepared"):
            data = report["routes"][workload][route]
            print(
                f"  {route:19s} {data['median_us']:.3f} us  "
                f"p10={data['p10_us']:.3f} us  "
                f"CV={data['cv'] * 100:.2f}%"
            )
        for route in ("curanddx_fresh", "curanddx_prepared"):
            print(
                f"  SofL/{route}: "
                f"{comparison[f'sofl_over_{route}_median_speedup']:.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
