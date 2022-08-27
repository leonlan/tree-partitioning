from collections import defaultdict
from time import perf_counter

import pyomo.environ as pyo

from .Result import Result


def single_stage(case, generators, time_limit):

    start = perf_counter()
    model, _ = single_stage_milp(case, generators, time_limit)
    end = perf_counter() - start

    return Result(
        case=case.name,
        n_clusters=len(generators),
        generator_sizes=[len(v) for v in generators.values()],
        power_flow_disruption=model.objective(),
        runtime=end,
        n_switched_lines=get_n_switched_lines(model),
        cluster_sizes=get_cluster_sizes(model),
        algorithm="Single stage",
    )


def single_stage_milp(case, generators, time_limit):
    """
    Solve the tree partitioning problem for transient stability using the
    single-stage MILP approach.
    """
    netdict = case.netdict
    buses, lines = netdict["buses"], netdict["lines"]
    clusters = generators

    # Define a model
    model = pyo.ConcreteModel("TPP-TS")

    # Declare decision variables
    model.assign_bus = pyo.Var(buses, clusters, domain=pyo.Binary)
    model.assign_line = pyo.Var(lines, clusters, domain=pyo.Binary)
    model.active_cross_edge = pyo.Var(lines, domain=pyo.Binary)
    model.active_line = pyo.Var(lines, domain=pyo.Binary)
    model.commodity_flow = pyo.Var(lines, domain=pyo.Reals)

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(lines[e]["f"] * (1 - m.active_line[e]) for e in lines)

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
        rhs1 = sum(m.assign_line[line, cluster] for cluster in clusters)
        rhs2 = m.active_cross_edge[line]
        lhs = m.active_line[line]
        return rhs1 + rhs2 == lhs

    @model.Constraint()
    def exactly_k_minus_1_cross_edges(m):
        return sum(m.active_cross_edge[line] for line in lines) == len(clusters) - 1

    @model.Constraint()
    def commodity_flow_source_bus(m):
        rhs1 = sum([m.commodity_flow[i, j, idx] for (i, j, idx) in lines if i == 0])
        rhs2 = sum([m.commodity_flow[j, i, idx] for (j, i, idx) in lines if i == 0])
        return rhs1 - rhs2 == len(buses) - 1

    @model.Constraint(buses)
    def commodity_flow_sink_buses(m, bus):
        if bus != 0:
            rhs1 = sum([m.commodity_flow[i, j, x] for (i, j, x) in lines if i == bus])
            rhs2 = sum([m.commodity_flow[j, i, x] for (j, i, x) in lines if i == bus])
            return rhs1 - rhs2 == -1

        return pyo.Constraint.Skip

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

    result = solver.solve(model, tee=False, options=options)

    # # Print solution
    # print(
    #     f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"
    # )

    return model, result


def get_cluster_sizes(model):
    cluster_sizes = defaultdict(int)

    for (_, cluster), val in model.assign_bus.items():
        if round(val()) == 1:
            cluster_sizes[cluster] += 1

    return list(cluster_sizes.values())


def get_n_switched_lines(model):
    res = 0

    for _, val in model.active_line.items():
        if round(val()) == 1:
            res += 1

    return res
