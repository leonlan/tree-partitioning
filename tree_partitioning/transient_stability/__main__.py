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
    parser.add_argument("--max_size", type=int, default=30)
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

    with open(args.results_path, "w") as fi:
        for path in instances:
            n = name2size(path)

            if n < args.min_size or n > args.max_size:
                continue

            try:
                case = Case.from_file(path, merge_lines=True)

                for k in range(2, args.n_clusters):
                    generator_groups = mst_gci(case, k)

                    try:
                        res1 = single_stage(case, generator_groups, args.time_limit)
                        fi.write(res1.to_csv())
                    except:
                        print(f"Failure: {case.name} - single stage")

                    try:
                        res2pfd = two_stage(
                            case,
                            generator_groups,
                            tpi_objective="power_flow_disruption",
                            time_limit=args.time_limit,
                        )
                        fi.write(res2pfd.to_csv())

                    except Exception as e:
                        print(f"Failure: {case.name} - two stage")
                        print(e)

                print(f"{case.name} finished")

            except:
                print(f"Failure: {path}")


if __name__ == "__main__":
    main()
