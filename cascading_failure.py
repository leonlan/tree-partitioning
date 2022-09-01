from collections import defaultdict
from pathlib import Path

import networkx as nx

import tree_partitioning.utils as utils
from tree_partitioning.classes import Case

from .constants import _EPS
from .dcpf import dcpf

# Which TP-network to choose
# IEEE-118
# Run another DCOPF ( might lead to bad situation bcs congestion=1 )
# - If TP-G congestion => 1, then run DCOPF
# - If not, then run both with and without DCOPF
#
# Variants of the problem
# - TP-TS: transient stability
# - TP-MC-DC: minimum congestion with dc power flows
# - Single stage / two-stage / recursive


class Statistics:
    """
    Store simulation statistics for cascading failures.
    """

    def __init__(self):
        self.lost_load: list[float] = []
        self.n_line_failures: list[int] = []
        self.n_gen_adjustments: list[int] = []
        self.n_alive_buses: list[int] = []
        self.n_final_components: list[int] = []

    def collect(
        self,
        lost_load,
        n_line_failures,
        n_gen_adjustments,
        n_alive_buses,
        n_final_components,
    ):
        self.lost_load.append(lost_load)
        self.n_line_failures.append(n_line_failures)
        self.n_gen_adjustments.append(n_gen_adjustments)
        self.n_alive_buses.append(n_alive_buses)
        self.n_final_components.append(n_final_components)

    def print_stats(self):

        print(
            f"{self.lost_load[-1]=:.2f}, {self.n_line_failures[-1]=},\
            {self.n_gen_adjustments[-1]=}, {self.n_alive_buses[-1]=}, {self.n_final_components[-1]=}"
        )


def cascading_failure(G):
    stats = Statistics()

    # Initiate a cascade for each possible line failure
    for line in G.edges:
        total_lost_load = 0
        n_line_failures = 0
        final_components = []

        components = utils.remove_lines(G, [line])
        overloaded = []

        for component in components:
            comp, lost_load = dcpf(component)
            total_lost_load += lost_load

            if congested_lines(comp):
                overloaded.append(comp)
            else:
                final_components.append(comp)

        while overloaded:
            new_components = []

            for component in overloaded:
                # Find the congested lines for each overloaded component
                lines = congested_lines(component)
                n_line_failures += len(lines)

                # Removing lines may create new components
                new_comps = utils.remove_lines(component, lines)

                # For every component, re-run DCPF and record lost load
                for comp in new_comps:
                    comp, lost_load = dcpf(comp)
                    total_lost_load += lost_load

                    new_components.append(comp)

            overloaded = []

            for comp in new_components:
                if congested_lines(comp):
                    overloaded.append(comp)
                else:
                    final_components.append(comp)

        # Collect post-cascading failure statistics
        n_gen_adjustments = adjusted_gens(G, final_components)
        n_alive_buses = len(G.nodes) - sum(
            [len(g.nodes) for g in final_components if total_generation(g) < _EPS]
        )
        n_final_components = len(final_components)

        stats.collect(
            total_lost_load,
            n_line_failures,
            n_gen_adjustments=n_gen_adjustments,
            n_alive_buses=n_alive_buses,
            n_final_components=n_final_components,
        )

        stats.print_stats()

    return stats


def total_generation(G):
    return sum(
        [-power for power in nx.get_node_attributes(G, "p_mw").values() if power < 0]
    )


def adjusted_gens(G, final_components):
    """
    Computes the number of adjusted generators given the initial graph G and
    the final components. We look at the original power injection and the
    post-CF power injections.
    """
    original = nx.get_node_attributes(G, "p_mw")

    # Dead buses by default have zero p_mw
    new = defaultdict(float)
    for component in final_components:
        new.update(nx.get_node_attributes(component, "p_mw"))

    adjustments = 0
    for bus in original.keys():
        if original[bus] < 0:  # Negative p_mw indicates generation
            adjustments += 1 if abs(original[bus] - new[bus]) > _EPS else 0

    return adjustments


def congested_lines(G):
    """
    Return the congested lines from G.
    """
    weights = nx.get_edge_attributes(G, "f")
    capacities = nx.get_edge_attributes(G, "c")

    congested = []

    for line in weights.keys():
        if abs(weights[line]) / capacities[line] > 1 + _EPS:
            congested.append(line)

    return congested


def dead_component(G):
    return False


def main():
    case = Case.from_file(
        # Path("instances/pglib_opf_case2000_goc.mat"), merge_lines=True
        # Path("instances/pglib_opf_case1888_rte.mat"),
        # Path("instances/pglib_opf_case2736sp_k.mat", merge_lines=True),
        Path("instances/pglib_opf_case118_ieee.mat", merge_lines=True),
        # Path("instances/pglib_opf_case300_ieee.mat", merge_lines=True),
        # Path("instances/pglib_opf_case500_goc.mat", merge_lines=True),
    )

    cascading_failure(case.G)


if __name__ == "__main__":
    main()
