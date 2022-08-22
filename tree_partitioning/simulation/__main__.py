# Initialize a network such that:
# Normal power injections
# Chen: Remain within the line limit and make it lower artificially using a multiplicative factor
# For TP: re-run OPF on new network (to make fair comparison)
from pathlib import Path

from tree_partitioning.classes import Case


class Statistics:
    def __init__(self):
        self.lost_load: float = 0
        self.n_line_failures: int = 0


def cascading_failure(net):
    case = Case.from_file(
        # Path("instances/pglib_opf_case2000_goc.mat"), merge_lines=True
        # Path("instances/pglib_opf_case1888_rte.mat"),
        # Path("instances/pglib_opf_case2736sp_k.mat", merge_lines=True),
        Path("instances/pglib_opf_case500_goc.mat", merge_lines=True),
    )

    stats = Statistics()

    while not stop(net):
        # Sample a single line or two lines
        # Remove the line(s)
        # Re-run PF
        # Collect stats
        pass

    return stats


def stop(net):
    # Measure: lost load until no more line will be failing/overloaded, number of failed lines
    pass
