import argparse
from collections import defaultdict
from glob import glob
from pathlib import Path

import _utils
import networkx as nx
import numpy as np
import pyomo.environ as pyo
from _single_stage_warm_start import _single_stage_warm_start
from _two_stage import _two_stage

import tree_partitioning.milp.partitioning as partitioning
import tree_partitioning.milp.tree_partitioning as single_stage
import tree_partitioning.utils as utils
from tree_partitioning.classes import Case
from tree_partitioning.constants import _EPS
from tree_partitioning.dcopf import dcopf
from tree_partitioning.dcopf_pp import dcopf_pp
from tree_partitioning.dcpf import dcpf
from tree_partitioning.gci import mst_gci
from tree_partitioning.milp.line_switching import (
    maximum_congestion as ls_maximum_congestion,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_pattern", default="instances/pglib_opf_*.mat")
    parser.add_argument("--max_clusters", type=int, default=4)
    parser.add_argument("--min_size", type=int, default=30)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--gci_weight", type=str, default="neg_weight")
    parser.add_argument("--results_dir", type=str, default="results/cfs/")

    return parser.parse_args()


class Statistics:
    """
    Store simulation statistics for cascading failures.
    """

    def __init__(self, case, G, name, n_clusters):
        # General stats
        self.case = case
        self.G = G
        self.name = name
        self.n_buses = len(G.nodes())
        self.n_lines = len(G.edges())
        self.n_clusters = n_clusters
        self._edge_weights = nx.get_edge_attributes(G, "f")

        # Cascading failure stats
        self.lines: list[tuple] = []
        self.lines_flow: list[tuple] = []
        self.lost_load: list[float] = []
        self.n_line_failures: list[int] = []
        self.n_gen_adjustments: list[int] = []
        self.n_alive_buses: list[int] = []
        self.n_final_components: list[int] = []

    def collect(
        self,
        line,
        lost_load,
        n_line_failures,
        n_gen_adjustments,
        n_alive_buses,
        n_final_components,
    ):
        self.lines.append((line, round(self._edge_weights[line])))
        self.lost_load.append(lost_load)
        self.n_line_failures.append(n_line_failures)
        self.n_gen_adjustments.append(n_gen_adjustments)
        self.n_alive_buses.append(n_alive_buses)
        self.n_final_components.append(n_final_components)

    def to_csv(self, path):
        with open(path, "w") as fi:
            for idx in range(len(self.lost_load)):
                fi.write(
                    "; ".join(
                        str(val)
                        for val in [
                            # Instance description
                            self.case.name,
                            self.name,
                            self.n_buses,
                            self.n_lines,
                            self.n_clusters,
                            # CF stats
                            self.lines[idx],
                            self.lost_load[idx],
                            self.n_line_failures[idx],
                            self.n_gen_adjustments[idx],
                            self.n_alive_buses[idx],
                            self.n_final_components[idx],
                        ]
                    )
                )
                fi.write("\n")


def cascading_failure(stats, G, export_path="tmp-cf.txt"):
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

            if utils.congested_lines(comp):
                overloaded.append(comp)
            else:
                final_components.append(comp)

        while overloaded:
            new_components = []

            for component in overloaded:
                # Find the congested lines for each overloaded component
                lines = utils.congested_lines(component)
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
                if utils.congested_lines(comp):
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
            line,
            total_lost_load,
            n_line_failures,
            n_gen_adjustments=n_gen_adjustments,
            n_alive_buses=n_alive_buses,
            n_final_components=n_final_components,
        )

    print(export_path)
    print(f"Lost load: {np.mean(stats.lost_load)}")
    print("\n")
    stats.to_csv(export_path)

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


def main():
    args = parse_args()
    instances = sorted(glob(args.instance_pattern), key=_utils.name2size)

    for path in instances:
        n = utils.name2size(path)

        if n < args.min_size or n > args.max_size:
            continue

        case = Case.from_file(path)
        Path(args.results_dir).mkdir(exist_ok=True, parents=True)

        # Original network
        name = f"{case.name}-original-{args.gci_weight}"
        stats = Statistics(case, case.G, name, 1)
        cascading_failure(stats, case.G, f"{args.results_dir}{name}.txt")
        solver = pyo.SolverFactory("gurobi", solver_io="python")
        options = {"TimeLimit": 60}

        for k in range(2, args.max_clusters + 1):
            generators = mst_gci(case, k, weight=args.gci_weight)

            name = f"{case.name}-ws{k}-{args.gci_weight}"
            try:
                partition, lines, runtime = _single_stage_warm_start(
                    case,
                    generators,
                    line_switching_model=ls_maximum_congestion,
                    solver=solver,
                    options=options,
                )

                post_G_dcopf = dcopf_pp(case.G, case.net.deepcopy(), lines)

                # k-TP'd OPF network
                stats = Statistics(case, post_G_dcopf, name, k)
                cascading_failure(stats, post_G_dcopf, f"{args.results_dir}{name}.txt")

            except:
                name = f"{case.name}-2st{k}-{args.gci_weight}"
                try:
                    partition, lines, runtime = _two_stage(
                        case,
                        generators,
                        partitioning_model=partitioning.power_flow_disruption,
                        line_switching_model=ls_maximum_congestion,
                        solver=solver,
                        options=options,
                    )
                    post_G_dcopf = dcopf_pp(case.G, case.net.deepcopy(), lines)
                    # k-TP'd OPF network
                    stats = Statistics(case, post_G_dcopf, name, k)
                    cascading_failure(
                        stats, post_G_dcopf, f"{args.results_dir}{name}.txt"
                    )
                    print(name, "failed")

                except:
                    print(name, "failed")


if __name__ == "__main__":
    main()
