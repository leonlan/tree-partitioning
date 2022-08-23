# Initialize a network such that:
# Normal power injections
# Chen: Remain within the line limit and make it lower artificially using a multiplicative factor
# For TP: re-run OPF on new network (to make fair comparison)
from pathlib import Path

import networkx as nx
import numpy as np
import pyomo.environ as pyo

from tree_partitioning.classes import Case

_EPS = 0.01


class Statistics:
    def __init__(self):
        self.lost_load: float = 0
        self.n_line_failures: int = 0


def cascading_failure(G):
    all_stats = []

    # Initiate a cascade for each possible line failure
    for line in G.edges:
        stats = Statistics()

        components = remove_lines(G, [line])
        overloaded = []

        for component in components:
            comp, load_shedding = dcpf(component)
            stats.lost_load += load_shedding

            if congested_lines(comp):
                overloaded.append(comp)

        while overloaded:
            new_overloaded = []

            for component in overloaded:
                # Find the congested lines for each overloaded component
                lines = congested_lines(component)
                stats.n_line_failures += len(lines)

                # Removing lines may create new components
                comps = remove_lines(component, lines)

                # For every component, re-run DCPF and record lost load
                for comp in comps:
                    comp, lost_load = dcpf(comp)

                    if not dead_component(comp):
                        new_overloaded.append(comp)

                    stats.lost_load += lost_load

            overloaded = [comp for comp in new_overloaded if congested_lines(comp)]

        print(f"{stats.n_line_failures=}, {stats.lost_load=:.2f}")
        all_stats.append(stats)

    print(f" Average line failure: {np.mean([s.n_line_failures for s in all_stats])}")
    print(f"Average load shedding: {np.mean([s.lost_load for s in all_stats])}")

    return all_stats


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
        # Path("instances/pglib_opf_case118_ieee.mat", merge_lines=True),
        Path("instances/pglib_opf_case300_ieee.mat", merge_lines=True),
    )

    cascading_failure(case.G)


if __name__ == "__main__":
    main()
