import numpy as np
import pyomo.environ as pyo

from tree_partitioning.classes import (
    Case,
    Partition,
    ReducedGraph,
    Solution,
)


def milp_line_switching(partition: Partition, objective="congestion"):
    """
    Solves the Line Switching Problem using MILP
    and returns the corresponding Tree Partition.
    """
    model, result = _milp_solve_pyomo(partition, objective=objective)

    if result.solver.termination_condition != "infeasible":
        # MILP integral variables may be non-integral in the end result
        # due to LP relaxations. We use np.isclose to extract the binary variables
        # for switching cross edges, and assert that the number of kept lines
        # are indeed correct.
        switched_lines = [
            tuple(line)
            for (_, _, *line), v in model.y.items()
            if np.isclose(0, v(), atol=1e-02)
        ]
        kept_lines = [
            tuple(line)
            for (_, _, *line), v in model.y.items()
            if np.isclose(1, v(), atol=1e-02)
        ]

        assert len(switched_lines) + len(kept_lines) == len(
            [_ for _ in model.y.items()]
        )
        assert len(kept_lines) == len(partition) - 1

        return Solution(partition, switched_lines, model=model)

    elif result.solver.termination_condition == "infeasible":
        raise "Infeasible"
    # TODO: remove this in the future and raise error
    else:
        return Solution(partition, [], model=model)


def _milp_solve_pyomo(partition: Partition, objective: str = "congestion"):
    """
    Pyomo model for LSP.
    """
    case = Case()
    netdict = case.netdict
    buses, lines = netdict["buses"], netdict["lines"]
    reduced_graph = ReducedGraph(case.G, partition)
    clusters = reduced_graph.clusters
    cross_edges = reduced_graph.cross_edges
    _cross_edge_lines = set(line for u, v, line in cross_edges)
    internal_edges = [
        line for line in netdict["lines"].keys() if line not in _cross_edge_lines
    ]
    ep2e = reduced_graph.cross_edge_to_line
    k = len(clusters)

    # Define a model
    model = pyo.ConcreteModel(f"Line Switching Problem: Minimize {objective}")

    # Declare decision variables
    model.gamma = pyo.Var(domain=pyo.NonNegativeReals)
    model.fabs = pyo.Var(lines, domain=pyo.NonNegativeReals)
    model.fp = pyo.Var(lines, domain=pyo.NonNegativeReals)
    model.fm = pyo.Var(lines, domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(buses, domain=pyo.Reals)
    model.y = pyo.Var(cross_edges, domain=pyo.Binary)
    model.q = pyo.Var(cross_edges, domain=pyo.Reals)
    M = {line: 10 * data["c"] for line, data in lines.items()}
    # M = {line: min(10000, max(20 * data["c"], 500)) for line, data in lines.items()}

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(m):
        if objective == "congestion":
            return model.gamma
        else:
            raise ValueError(f"Objective {objective} is not valid.")

    # Declare constraints
    """
    Maximum congestion and auxilliary expressions
    """

    @model.Constraint(lines)
    def max_congestion_lower_bound(m, *e):
        return m.gamma >= m.fabs[e] / lines[e]["c"]

    @model.Constraint(lines)
    def absolute_flows(m, *e):
        return m.fabs[e] == m.fp[e] + m.fm[e]

    @model.Expression(lines)
    def real_flow(m, *e):
        return m.fp[e] - m.fm[e]

    @model.Expression(lines)
    def dc_flow(m, *e):
        i, j, idx = e
        return lines[e]["b"] * (m.theta[i] - m.theta[j])

    """
    Reduced graph constraints
    """

    @model.Expression(clusters)
    def incoming_and_outgoing_cluster_flow(m, r):
        return sum(
            sign * m.q[u, v, line] for (u, v, line), sign in reduced_graph.incidence(r)
        )

    @model.Constraint(clusters)
    def commodity_flow_conservation(m, r):
        if r == 0:
            return m.incoming_and_outgoing_cluster_flow[r] == k - 1
        else:
            return m.incoming_and_outgoing_cluster_flow[r] == -1

    @model.Constraint(cross_edges)
    def commodity_flow_upper_bound(m, *ce):
        return m.q[ce] <= (k - 1) * m.y[ce]

    @model.Constraint(cross_edges)
    def commodity_flow_lower_bound(m, *ce):
        return m.q[ce] >= -(k - 1) * m.y[ce]

    @model.Constraint()
    def reduced_graph_spanning_tree(m):
        return sum(m.y[ce] for ce in cross_edges) == k - 1

    """
    Relate cross edges and line switching actions
    """

    # FIXME: For some reason, cross_edges elements are flattened tuples
    # (u, v, i, j, k) instead of (u, v, (i, j, k))
    @model.Constraint(cross_edges)
    def active_cross_edge_1(m, *ce):
        e = tuple(ce[2:])
        return m.real_flow[e] >= m.dc_flow[e] - (1 - m.y[ce]) * M[e]

    @model.Constraint(cross_edges)
    def active_cross_edge_2(m, *ce):
        e = tuple(ce[2:])
        return m.real_flow[e] <= m.dc_flow[e] + (1 - m.y[ce]) * M[e]

    @model.Constraint(cross_edges)
    def inactive_cross_edge_1(m, *ce):
        e = tuple(ce[2:])
        return m.real_flow[e] >= -m.y[ce] * M[e]

    @model.Constraint(cross_edges)
    def inactive_cross_edge_2(m, *ce):
        e = tuple(ce[2:])
        return m.real_flow[e] <= m.y[ce] * M[e]

    """
    Regular DC power flows
    """

    @model.Constraint(internal_edges)
    def internal_edges_flow(m, *e):
        return m.real_flow[e] == m.dc_flow[e]

    @model.Expression(buses)
    def outgoing_bus_flow(m, i):
        return sum(m.fp[e] - m.fm[e] for e in lines if e[0] == i)

    @model.Expression(buses)
    def incoming_bus_flow(m, i):
        return sum(m.fp[e] - m.fm[e] for e in lines if e[1] == i)

    @model.Constraint(buses)
    def flow_conservation(m, i):
        return m.outgoing_bus_flow[i] - m.incoming_bus_flow[i] == buses[i]["p_mw"]

    @model.Constraint()
    def ref_bus(m):
        return m.theta[0] == 0

    # Solve
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    result = solver.solve(model, tee=True, options={"TimeLimit": 300})

    # Print solution
    print(
        f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"
    )
    return model, result
