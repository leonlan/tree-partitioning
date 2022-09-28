from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import _utils
import pyomo.environ as pyo
from _recursive import _recursive
from _single_stage import _single_stage
from _single_stage_warm_start import _single_stage_warm_start
from _two_stage import _two_stage
from Result import make_result

import tree_partitioning.line_switching.brute_force as brute_force
import tree_partitioning.milp.line_switching.maximum_congestion as milp_line_switching
import tree_partitioning.milp.partitioning as partitioning
import tree_partitioning.milp.tree_partitioning as single_stage
from tree_partitioning.classes import Case
from tree_partitioning.gci import mst_gci


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_pattern", default="instances/pglib_opf_*.mat")
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--min_clusters", type=int, default=2)
    parser.add_argument("--max_clusters", type=int, default=3)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--min_size", type=int, default=30)
    parser.add_argument(
        "--algorithm",
        choices=["single_stage", "two_stage", "warm_start"],
        default="single_stage two_stage warm_start",
        nargs="+",
    )
    parser.add_argument("--results_dir", type=str, default="results/mc/")
    parser.add_argument("--gci_weight", type=str, default="neg_weight")

    return parser.parse_args()


def setup_config(args):
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    options = {"TimeLimit": args.time_limit}
    config = {"solver": solver, "options": options}

    return config


def main():
    args = parse_args()
    instances = sorted(glob(args.instance_pattern), key=_utils.name2size)
    config = setup_config(args)

    Path(args.results_dir).mkdir(exist_ok=True, parents=True)

    for path in instances:

        if not (args.min_size <= _utils.name2size(path) <= args.max_size):
            continue

        case = Case.from_file(path, merge_lines=True)
        print(case.name)

        for k in range(args.min_clusters, args.max_clusters + 1):
            generator_groups = mst_gci(case, k, weight=args.gci_weight)

            if "single_stage" in args.algorithm:
                path = f"{args.results_dir}{case.name}-1ST-{k}.csv"
                try:
                    partition, lines, runtime = _single_stage(
                        case,
                        generator_groups,
                        tree_partitioning_alg=single_stage.maximum_congestion,
                        **config,
                    )

                    make_result(
                        case,
                        generator_groups,
                        partition,
                        lines,
                        runtime=runtime,
                        algorithm="1ST",
                    ).to_csv(path)
                except Exception as e:
                    print(path, e)

            if "two_stage" in args.algorithm:
                path = f"{args.results_dir}{case.name}-2ST-{k}.csv"
                try:
                    partition, lines, runtime = _two_stage(
                        case,
                        generator_groups,
                        partitioning_model=partitioning.power_flow_disruption,
                        line_switching_model=milp_line_switching,
                        **config,
                    )
                    make_result(
                        case,
                        generator_groups,
                        partition,
                        lines,
                        runtime=runtime,
                        algorithm="2ST",
                    ).to_csv(path)

                except:
                    print(path, "failed")

            if "warm_start" in args.algorithm:
                try:
                    path = f"{args.results_dir}{case.name}-ws-{k}.csv"
                    partition, lines, runtime = _single_stage_warm_start(
                        case,
                        generator_groups,
                        line_switching_model=milp_line_switching,
                        **config,
                    )
                    make_result(
                        case,
                        generator_groups,
                        partition,
                        lines,
                        runtime=runtime,
                        algorithm="ws",
                    ).to_csv(path)

                except:
                    print(path, "failed")


if __name__ == "__main__":
    main()
