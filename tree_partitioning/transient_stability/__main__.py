import argparse
import re
from glob import glob

from tree_partitioning.classes import Case
from tree_partitioning.gci import mst_gci

from .multi_stage import multi_stage
from .single_stage import single_stage
from .two_stage import two_stage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_pattern", default="instances/pglib_opf_*.mat")
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=1000)
    parser.add_argument("--min_size", type=int, default=30)

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

            print(f"Case {case.name}, {k=}")
            generator_groups = mst_gci(k)
            res1 = single_stage(case, generator_groups, args.time_limit)
            print(f"Single stage: {res1.power_flow_disruption}")

            res2pfd = two_stage(
                case,
                generator_groups,
                tpi_objective="power_flow_disruption",
                time_limit=args.time_limit,
            )

            print(f"Two stage: {res2pfd.power_flow_disruption}")

            res2pi = two_stage(
                case,
                generator_groups,
                tpi_objective="power_imbalance",
                time_limit=args.time_limit,
            )

            print(f"Two stage: {res2pi.power_flow_disruption}")

            # res_multi = multi_stage(
            #     case,
            #     generator_groups,
            #     time_limit=args.time_limit,
            # )

            # print(f"Multi-stage: {res_multi.power_flow_disruption}")


if __name__ == "__main__":
    main()
