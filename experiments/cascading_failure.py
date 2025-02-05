import argparse
import numpy as np
from collections import defaultdict
from functools import partial
from glob import glob
from pathlib import Path

import _utils
import networkx as nx
import pyomo.environ as pyo
from _single_stage import _single_stage
from _single_stage_warm_start import _single_stage_warm_start
from _two_stage import _two_stage
from numpy.testing import assert_almost_equal
from tqdm.contrib.concurrent import process_map

import tree_partitioning.milp.line_switching.maximum_congestion as milp_line_switching
from tree_partitioning.line_switching.maximum_spanning_tree import maximum_spanning_tree
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
    parser.add_argument("--instance_pattern", default="instances-rev/pglib_opf_*.mat")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument(
        "--method", type=str, default="single_stage_power_flow_disruption"
    )
    parser.add_argument("--gci_weight", type=str, default="neg_weight")
    parser.add_argument("--results_dir", type=str, default="results-rev")
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--milp_time_limit", type=int, default=300)

    return parser.parse_args()


class Statistics:
    """
    Store simulation statistics for cascading failures.
    """

    def __init__(self, G, name, method, gci_weight, n_clusters):
        # General stats
        self.G = G
        self.name = name
        self.method = method
        self.gci_weight = gci_weight
        self.n_clusters = n_clusters
        self.n_buses = len(G.nodes())
        self.n_lines = len(G.edges())
        self.initial_load = total_load(G)
        self._edge_weights = nx.get_edge_attributes(G, "f")

        # Cascading failure stats
        self.lines: list[tuple] = []
        self.lines_flow: list[tuple] = []
        self.total_lost_load: list[float] = []
        self.n_iterations: list[int] = []
        self.n_line_failures: list[int] = []
        self.n_gen_adjustments: list[int] = []
        self.n_alive_buses: list[int] = []
        self.n_final_components: list[int] = []

    def collect(
        self,
        line,
        total_lost_load,
        n_iterations,
        n_line_failures,
        n_gen_adjustments,
        n_alive_buses,
        n_final_components,
    ):
        self.lines.append((line, round(self._edge_weights[line])))
        self.total_lost_load.append(total_lost_load)
        self.n_iterations.append(n_iterations)
        self.n_line_failures.append(n_line_failures)
        self.n_gen_adjustments.append(n_gen_adjustments)
        self.n_alive_buses.append(n_alive_buses)
        self.n_final_components.append(n_final_components)

    def to_csv(self, path):
        with open(path, "w") as fi:
            for idx in range(len(self.total_lost_load)):
                fi.write(
                    "; ".join(
                        str(val)
                        for val in [
                            # Instance description
                            self.name,
                            self.method,
                            self.gci_weight,
                            self.n_clusters,
                            self.n_buses,
                            self.n_lines,
                            self.initial_load,
                            # CF stats
                            self.lines[idx],
                            self.total_lost_load[idx],
                            self.n_iterations[idx],
                            self.n_line_failures[idx],
                            self.n_gen_adjustments[idx],
                            self.n_alive_buses[idx],
                            self.n_final_components[idx],
                        ]
                    )
                )
                fi.write("\n")


def cascade_single_line_failure(G, line):
    """
    Initiate a cascading failure by removing the passed-in line from G.
    """
    n_iterations = 1

    components = utils.remove_lines(G, [line], copy=True)
    overloaded = []
    final_components = []

    for comp in components:
        proportional_control(comp)
        dcpf(comp)

        if utils.congested_lines(comp):
            overloaded.append(comp)
        else:
            final_components.append(comp)

    while overloaded:
        new_overloaded = []

        for component in overloaded:
            # Find the congested lines for each overloaded component
            lines = utils.congested_lines(component)

            # Removing lines may create new components
            new_components = utils.remove_lines(component, lines, copy=False)

            for comp in new_components:
                # Adjust generation or shed load using proportional control and re-run DCPF
                proportional_control(comp)
                dcpf(comp)

                if utils.congested_lines(comp):  # proxy for determining if overloaded
                    new_overloaded.append(comp)
                else:
                    final_components.append(comp)

        # Continue cascade on the new overloaded components
        overloaded = new_overloaded
        n_iterations += 1

    # Collect post-cascading failure statistics
    n_line_failures = len(G.edges) - sum(len(comp.edges) for comp in final_components)
    total_lost_load = total_load(G) - sum(total_load(comp) for comp in final_components)
    n_gen_adjustments = adjusted_gens(G, final_components)
    n_alive_buses = len(G.nodes) - sum(
        [len(g.nodes) for g in final_components if total_generation(g) < _EPS]
    )
    n_final_components = len(final_components)

    # Sanity checks
    assert sum(len(comp.nodes) for comp in final_components) == len(G.nodes)
    assert total_lost_load <= total_generation(G)

    return {
        "line": line,
        "total_lost_load": total_lost_load,
        "n_iterations": n_iterations,
        "n_line_failures": n_line_failures,
        "n_gen_adjustments": n_gen_adjustments,
        "n_alive_buses": n_alive_buses,
        "n_final_components": n_final_components,
    }


def cascading_failure(stats, G, max_workers=8):
    """
    Runs a cascading failure simulation on the network G in parallel.
    """
    tqdm_kwargs = dict(max_workers=max_workers, unit="instance")
    func = partial(cascade_single_line_failure, G)
    func_args = [line for line in G.edges]

    data = []
    for args in func_args:
        try:
            data.append(func(args))
        except:
            pass

    # data = process_map(func, func_args, **tqdm_kwargs)

    for res in data:
        if res is not None:
            stats.collect(**res)

    return stats


def proportional_control(G, in_place=True):
    """
    Readjusts generation or load to maintain power balance using a proportional
    control scheme.

    If there is a generation surplus, all generator outputs are lowered by the
    same proportion to match the load. If there is a load surplus, then all
    loads are adjusted (i.e., load shedding) to match the generation.

    If in_place is set to True, then the power adjustments are made in place.
    Otherwise, the graph G is copied.
    """
    H = G if in_place else G.copy()

    # No adjustments needed if there is no generation and no load,
    # or if there is no power imbalance.
    power_imbalance = get_power_imbalance(H)
    if max(total_generation(H), total_load(H)) == 0 or power_imbalance == 0:
        return H

    proportion = abs(power_imbalance) / max(total_generation(H), total_load(H))

    if power_imbalance > 0:  # load surplus
        old_load = nx.get_node_attributes(H, "p_load_total").items()
        new_load = {bus: (1 - proportion) * load for bus, load in old_load}
        nx.set_node_attributes(H, new_load, "p_load_total")
    else:  # generation surplus
        old_gen = nx.get_node_attributes(H, "p_gen_total").items()
        new_gen = {bus: (1 - proportion) * gen for bus, gen in old_gen}
        nx.set_node_attributes(H, new_gen, "p_gen_total")

    # Power imbalance on the adjusted graph should be near-zero.
    assert_almost_equal(get_power_imbalance(H), 0)
    return H


def total_generation(G):
    return sum(nx.get_node_attributes(G, "p_gen_total").values())


def total_load(G):
    return sum(nx.get_node_attributes(G, "p_load_total").values())


def get_power_imbalance(G):
    """
    Return the power imbalance. Positive power imbalance means that
    there is more load than generation.
    """
    return total_load(G) - total_generation(G)


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

    res_dir = Path(args.results_dir)
    res_dir.mkdir(exist_ok=True, parents=True)

    for path in instances:
        case = Case.from_file(path)

        if args.method == "original":
            stats = Statistics(case.G, case.name, args.method, args.gci_weight, 1)
            cascading_failure(stats, case.G, args.num_procs)

            path = res_dir / f"{case.name}-{args.method}-{args.gci_weight}-k1.csv"
            stats.to_csv(path)
            continue

        # Tree partitioning methods
        k = args.n_clusters
        generators = mst_gci(case, k, weight=args.gci_weight)

        solver = pyo.SolverFactory("gurobi", solver_io="python")
        options = {"TimeLimit": args.milp_time_limit}

        try:
            if args.method == "single_stage_power_flow_disruption":
                _, lines, _ = _single_stage(
                    case,
                    generators,
                    tree_partitioning_alg=single_stage.power_flow_disruption,
                    solver=solver,
                    options=options,
                )
            elif args.method == "single_stage_maximum_congestion":
                _, lines, _ = _single_stage(
                    case,
                    generators,
                    tree_partitioning_alg=single_stage.maximum_congestion,
                    solver=solver,
                    options=options,
                )
            elif args.method == "two_stage_maximum_congestion":
                _, lines, _ = _two_stage(
                    case,
                    generators,
                    partitioning_model=partitioning.power_flow_disruption,
                    line_switching_model=milp_line_switching,
                    solver=solver,
                    options=options,
                )
            elif args.method == "two_stage_power_flow_disruption":
                _, lines, _ = _two_stage(
                    case,
                    generators,
                    partitioning_model=partitioning.power_flow_disruption,
                    line_switching_alg=maximum_spanning_tree,
                    solver=solver,
                    options=options,
                )

            elif args.method == "warm_start_maximum_congestion":
                _, lines, _ = _single_stage_warm_start(
                    case,
                    generators,
                    partitioning_model=partitioning.power_flow_disruption,
                    line_switching_model=milp_line_switching,
                    solver=solver,
                    options=options,
                )

            # Take the resulting line switching actions and create a new
            # network. Run DCOPF using pandapower afterwards.
            post_G_dcopf = dcopf_pp(case.G, case.net.deepcopy(), lines)

            stats = Statistics(post_G_dcopf, case.name, args.method, args.gci_weight, k)
            cascading_failure(stats, post_G_dcopf, args.num_procs)
            print(np.mean(stats.total_lost_load))

            path = res_dir / f"{case.name}-{args.method}-{args.gci_weight}-k{k}.csv"
            stats.to_csv(path)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
