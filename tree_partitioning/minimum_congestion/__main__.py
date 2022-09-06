import argparse
import re
from glob import glob

import pyomo.environ as pyo

import tree_partitioning.utils as utils
from tree_partitioning.classes import Case
from tree_partitioning.gci import mst_gci
from tree_partitioning.line_switching.milp_line_switching import milp_line_switching
from tree_partitioning.partitioning import milp_cluster, model2partition
from tree_partitioning.single_stage.minimum_congestion import (
    minimum_congestion as single_stage,
)

from .recursive import recursive
from .two_stage import two_stage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_pattern", default="instances/pglib_opf_*.mat")
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=100)
    parser.add_argument("--min_size", type=int, default=30)
    parser.add_argument("--results_path", type=str, default="results.txt")

    return parser.parse_args()


def main():
    args = parse_args()
    instances = sorted(glob(args.instance_pattern), key=utils.name2size)

    for path in instances:
        n = utils.name2size(path)

        if n < args.min_size or n > args.max_size:
            continue

        case = Case.from_file(path, merge_lines=True)

        for k in range(2, args.n_clusters + 1):
            print(f"{case.name=} and {k=}")

            # try:
            generator_groups = mst_gci(case, k)

            solver = pyo.SolverFactory("gurobi", solver_io="python")
            model = single_stage(case, generator_groups, 100)
            res = solver.solve(
                model, tee=True, options={"FeasibilityTol": 0.01, "MIPFocus": 2}
            )
            print(f"test: {model.objective()}")

            # except Exception as e:
            #     print(e)


if __name__ == "__main__":
    main()
