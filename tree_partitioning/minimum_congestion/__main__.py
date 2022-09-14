import argparse
import re
from glob import glob

import pyomo.environ as pyo

import tree_partitioning.utils as utils
from tree_partitioning.classes import Case
from tree_partitioning.gci import mst_gci
from tree_partitioning.single_stage import maximum_congestion as single_stage
from tree_partitioning.single_stage import transient_stability as single_stage_ts


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

            # Solve TS first
            model_ts = single_stage_ts(case, generator_groups)
            res_ts = solver.solve(
                model_ts, tee=True, options={"FeasibilityTol": 0.01, "MIPFocus": 2}
            )

            # Feed solution TS into MC
            model = single_stage(case, generator_groups)

            for key, value in model_ts.assign_bus.items():
                model.assign_bus[key] = round(value())

            for key, value in model_ts.assign_line.items():
                model.assign_line[key] = round(value())

            for key, value in model_ts.active_cross_edge.items():
                model.active_cross_edge[key] = round(value())

            for key, value in model_ts.active_line.items():
                model.active_line[key] = round(value())

            for key, value in model_ts.commodity_flow.items():
                model.commodity_flow[key] = round(value())

            res = solver.solve(
                model,
                tee=True,
                options={"FeasibilityTol": 0.01, "MIPFocus": 1},
                warmstart=True,
            )
            print(f"test: {model.objective()}")

            # except Exception as e:
            #     print(e)


if __name__ == "__main__":
    main()
