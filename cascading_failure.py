# Initialize a network such that:
# Normal power injections
# Chen: Remain within the line limit and make it lower artificially using a multiplicative factor
# For TP: re-run OPF on new network (to make fair comparison)
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pyomo.environ as pyo

from tree_partitioning.classes import Case

_EPS = 0.001

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

        components = remove_lines(G, [line])
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
                new_comps = remove_lines(component, lines)

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


def remove_lines(G, lines):
    """
    Remove the passed-in lines from G. Return all connected components.
    """
    H = G.copy()
    H.remove_edges_from(lines)
    return [H.subgraph(c).copy() for c in nx.weakly_connected_components(H)]


def dead_component(G):
    return False


def dcpf(G):
    """
    Solve DC power flow for the passed-in graph G with possibly load shedding
    or generation adjustments.

    Return a new graph with adjusted power flows and the load shedding.
    """
    buses, lines = G.nodes, G.edges

    # Define a model
    model = pyo.ConcreteModel("DC-PF with adjustments")

    # Define variables
    model.gen_adjustment = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.load_shedding = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.theta = pyo.Var(buses, domain=pyo.Reals)
    model.flow = pyo.Var(lines, domain=pyo.Reals)

    # Either generation loss or load shedding
    # Positive power imbalance indicates more load than generation
    power_imbalance = sum(nx.get_node_attributes(G, "p_mw").values())

    if power_imbalance >= 0:
        model.no_gen_adj = pyo.Constraint(expr=model.gen_adjustment == 1)
    else:
        model.no_load_shedding = pyo.Constraint(expr=model.load_shedding == 1)

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(m):
        """
        Minimize the gen adjustment or load shedding needed to get to a
        convergent solution.
        """
        return m.gen_adjustment + m.load_shedding

    # Declare constraints
    @model.Expression(buses)
    def outgoing_flow(m, bus):
        return sum(m.flow[line] for line in lines if line[0] == bus)

    @model.Expression(buses)
    def incoming_flow(m, bus):
        return sum(m.flow[line] for line in lines if line[1] == bus)

    @model.Constraint(buses)
    def flow_conservation(m, bus):
        lhs = m.outgoing_flow[bus] - m.incoming_flow[bus]
        is_gen = buses[bus]["p_mw"] < 0

        if is_gen:
            rhs = m.gen_adjustment * buses[bus]["p_mw"]
        else:
            rhs = m.load_shedding * buses[bus]["p_mw"]

        return lhs == rhs

    @model.Constraint(lines)
    def susceptance(m, *line):
        i, j, _ = line
        return m.flow[line] == lines[line]["b"] * (m.theta[i] - m.theta[j])

    # Solve
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    result = solver.solve(model, tee=False, options={"TimeLimit": 300})

    # # Print solution
    # print(f"{model.gen_adjustment.value=:.2f}, {model.load_shedding.value=:.2f}")

    # Make a new copy of the graph and change the power injections and flows
    H = G.copy()

    new_power_injections = {
        bus: (model.load_shedding.value if p_mw > 0 else model.gen_adjustment.value)
        * p_mw
        for bus, p_mw in nx.get_node_attributes(H, "p_mw").items()
    }
    nx.set_node_attributes(H, new_power_injections, "p_mw")

    new_flows = {k: v.value for k, v in model.flow.items()}
    nx.set_edge_attributes(H, new_flows, "f")

    return H, (power_imbalance if power_imbalance > 0 else 0)


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