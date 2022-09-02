import argparse
import re
from glob import glob

from tree_partitioning.classes import Case
from tree_partitioning.gci import mst_gci
from tree_partitioning.line_switching.milp_line_switching import milp_line_switching
from tree_partitioning.partitioning import milp_cluster, model2partition

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


def name2size(name: str) -> int:
    """
    Extracts the instance size (i.e., num clients) from the instance name.
    """
    return int(re.search(r"_case(\d+)", name).group(1))


def main():
    args = parse_args()
    instances = sorted(glob(args.instance_pattern), key=name2size)

    for path in instances:
        n = name2size(path)

        if n < args.min_size or n > args.max_size:
            continue

        case = Case.from_file(path, merge_lines=True)

        for k in range(2, args.n_clusters):

            generator_groups = mst_gci(case, k)
            twostage_pfd = two_stage(
                case,
                n_clusters=k,
                partitioning_alg=lambda: model2partition(
                    milp_cluster(
                        case, generator_groups, "power_flow_disruption", args.time_limit
                    )[0]
                ),
                line_switching_alg=milp_line_switching,
            )

            twostage_pi = two_stage(
                case,
                n_clusters=k,
                partitioning_alg=lambda: model2partition(
                    milp_cluster(
                        case, generator_groups, "power_imbalance", args.time_limit
                    )[0]
                ),
                line_switching_alg=milp_line_switching,
            )


if __name__ == "__main__":
    main()
