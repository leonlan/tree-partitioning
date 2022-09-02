from collections import defaultdict

import networkx as nx
import pyomo.environ as pyo

from tree_partitioning.classes import Case, Partition, ReducedGraph


def milp_cluster(case, generators, objective, time_limit):
    """
    Solve the Tree Partition Identification minimizing transient stability.
    """
    buses, lines = case.G.nodes, case.G.edges
    clusters = generators

    # Define a model
    model = pyo.ConcreteModel("TPI-TS")

    # Declare decision variables
    model.assign_bus = pyo.Var(buses, clusters, domain=pyo.Binary)
    model.assign_line = pyo.Var(lines, clusters, domain=pyo.Binary)
    model.active_line = pyo.Var(lines, domain=pyo.Binary)
    model.commodity_flow = pyo.Var(lines, domain=pyo.Reals)
    model.cluster_size = pyo.Var(clusters, domain=pyo.Integers)
    model.power_imbalance = pyo.Var(clusters, domain=pyo.NonNegativeReals)

    @model.Expression(clusters)
    def total_power_imbalance(m, cluster):
        return sum(
            data["p_mw"] * m.assign_bus[bus, cluster] for bus, data in buses.items()
        )

    @model.Constraint(clusters)
    def lower_bound_power_imbalance(m, cluster):
        return m.power_imbalance[cluster] >= m.total_power_imbalance[cluster]

    @model.Constraint(clusters)
    def upper_bound_power_imbalance(m, cluster):
        return -m.power_imbalance[cluster] <= m.total_power_imbalance[cluster]

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(m):
        if objective == "power_flow_disruption":
            return sum(lines[e]["f"] * (1 - m.active_line[e]) for e in lines)
        else:
            return sum(m.power_imbalance[r] for r in clusters)

    # Declare constraints
    @model.Constraint(buses)
    def assign_generator_to_corresponding_cluster(m, bus):
        for cluster, nodes in generators.items():
            if bus in nodes:
                return m.assign_bus[bus, cluster] == 1

        return pyo.Constraint.Skip

    @model.Constraint(buses)
    def assign_bus_to_exactly_one_cluster(m, bus):
        return sum(m.assign_bus[bus, cluster] for cluster in clusters) == 1

    @model.Constraint(clusters, lines)
    def assign_line_only_if_assign_bus_i(m, cluster, *line):
        i, j, _ = line
        return m.assign_line[line, cluster] <= m.assign_bus[i, cluster]

    @model.Constraint(clusters, lines)
    def assign_line_only_if_assign_bus_j(m, cluster, *line):
        i, j, _ = line
        return m.assign_line[line, cluster] <= m.assign_bus[j, cluster]

    @model.Constraint(clusters, lines)
    def assign_line_if_both_assign_buses(m, cluster, *line):
        i, j, _ = line
        lhs = m.assign_line[line, cluster]
        rhs = m.assign_bus[i, cluster] + m.assign_bus[j, cluster] - 1
        return lhs >= rhs

    @model.Constraint(lines)
    def relate_assignment_to_activeness(m, *line):
        lhs = sum(m.assign_line[line, cluster] for cluster in clusters)
        rhs = m.active_line[line]

        return lhs == rhs

    @model.Constraint(clusters)
    def determine_cluster_size(m, cluster):
        lhs = sum(m.assign_bus[bus, cluster] for bus in buses)

        return lhs == m.cluster_size[cluster]

    @model.Constraint(clusters)
    def commodity_flow_source_bus(m, cluster):
        # Take the first generator in the generator group
        source = generators[cluster][0]

        out = [m.commodity_flow[i, j, idx] for (i, j, idx) in lines if i == source]
        inc = [m.commodity_flow[j, i, idx] for (j, i, idx) in lines if i == source]

        return sum(out) - sum(inc) == m.cluster_size[cluster] - 1

    @model.Constraint(buses)
    def commodity_flow_sink_buses(m, bus):
        # Skip all source generator nodes
        if bus in [gens[0] for _, gens in generators.items()]:
            return pyo.Constraint.Skip

        rhs1 = sum([m.commodity_flow[i, j, x] for (i, j, x) in lines if i == bus])
        rhs2 = sum([m.commodity_flow[j, i, x] for (j, i, x) in lines if i == bus])

        return rhs1 - rhs2 == -1

    @model.Constraint(lines)
    def commodity_flow_only_if_active_lhs(m, *line):
        return -(len(buses) - 1) * m.active_line[line] <= m.commodity_flow[line]

    @model.Constraint(lines)
    def commodity_flow_only_if_active_rhs(m, *line):
        return m.commodity_flow[line] <= (len(buses) - 1) * m.active_line[line]

    # Solve
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    options = {}
    options["TimeLimit"] = time_limit

    if objective == "power_imbalance":
        options["MIPGapAbs"] = 20

    result = solver.solve(model, tee=False, options=options)

    # Print solution
    # print(
    #     f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"
    # )

    return model, result


def model2partition(model):
    partition = defaultdict(list)

    for x, value in model.assign_bus.items():
        # NOTE Rounding values because solver returns non-integral vals
        if round(value()) == 1:
            bus, cluster = x
            partition[cluster].append(bus)

    return Partition(partition)
