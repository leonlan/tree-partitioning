#!/usr/bin/env ipython
import pyomo.environ as pyo

from tree_partitioning.classes import Case, TreePartition, Partition, SwitchedLines


def milp_line_switching(partition: Partition, objective="congestion"):
    """
    Solves the Line Switching Problem using MILP
    and returns the corresponding Tree Partition.

    Args:
    - G: NetworkX graph
    - P: Partition object
    - objective: Objective function to be minimized
        - "congestion": maximum congestion
        - "pfd": power flow disruption
    - solver (optional): Choice of solver defined by the pulp api
    """

    switched_lines = SwitchedLines(...)
    return TreePartition(partition, switched_lines)


def _milp_pulp(G, P, objective="gamma", solver=""):
    """Solves the OBS problem as MILP.

    The core idea of Optimal Bridge Switching (OBS) is to select a spanning tree
    in the reduced graph G_P. The lines that were not selected are to be switched
    off.

    Args:
    - G: NetworkX graph
    - objective: Objective function to be minimized
        - "gamma": maximum congestion
        - "pfd": power flow disruption
        - otherwise the MILP solves for feasibility
    - solver (optional): Choice of solver defined by the pulp api

    Returns:
    - results (dict): Dictionary containing all the interesting parameters.
        - ...
        - ...
    """
    GP = reduced_graph(G, P)
    vertex_ids = G.nodes
    edge_ids = G.edges
    edge_P_ids = GP.edges
    vertex_P_ids = GP.nodes

    name2e = {G.get_edge_data(*e)["name"]: e for e in edge_ids}
    name2ep = {GP.get_edge_data(*ep)["name"]: ep for ep in edge_P_ids}
    cross_edges = [(name2e[name], name2ep[name]) for name in name2ep.keys()]
    internal_edges = [
        name2e[name] for name in name2e.keys() if name not in name2ep.keys()
    ]

    # Parameters
    n = len(vertex_ids)
    p = {v: G.nodes[v]["p"] for v in G.nodes}
    b = {e: G.get_edge_data(*e)["b"] for e in G.edges}
    c = {e: G.get_edge_data(*e)["c"] for e in G.edges}
    M = {e: 100 * G.get_edge_data(*e)["c"] for e in G.edges}
    s = {v: (-1 if v != 0 else len(vertex_P_ids) - 1) for v in GP.nodes}
    weight = {e: GP.get_edge_data(*e)["weight"] for e in GP.edges}
    neighbors = incident_edges(G)
    neighbors_reduced = incident_edges(GP)

    ##########################################
    ####### Start MILP formulation ###########
    ##########################################

    # Variables
    gamma = pl.LpVariable("gamma", cat="Continuous")
    y = pl.LpVariable.dicts("y", edge_P_ids, cat="Binary")
    f = pl.LpVariable.dicts("f", edge_ids, cat="Continuous")
    fplus = pl.LpVariable.dicts("fplus", edge_ids, lowBound=0, cat="Continuous")
    fmin = pl.LpVariable.dicts("fmin", edge_ids, lowBound=0, cat="Continuous")
    q = pl.LpVariable.dicts("q", edge_P_ids, cat="Continuous")
    theta = pl.LpVariable.dicts("theta", vertex_ids, cat="Continuous")

    # Define the model
    model = pl.LpProblem("OBS", pl.LpMinimize)

    # Objective function
    if objective == "gamma":
        model += gamma
    elif objective == "pfd":
        model += pl.lpSum([weight[e] * y[e] for e in edge_P_ids])
    else:
        model += 0

    # Auxilliary max variable constraint
    for i, j, k in edge_ids:
        model += gamma >= f[(i, j, k)] * (1 / c[(i, j, k)])

    # Absolute flow
    for i, j, k in edge_ids:
        model += f[(i, j, k)] == fplus[(i, j, k)] + fmin[(i, j, k)]

    # SCF #1: Flow conservation
    for r in vertex_P_ids:
        model += (
            pl.lpSum([di * q[(u, v, w)] for (u, v, w), di in neighbors_reduced[r]])
            == s[r]
        )
    # SCF #2: Bounds
    for (u, v, w) in edge_P_ids:
        model += q[(u, v, w)] <= (len(vertex_P_ids) - 1) * y[(u, v, w)]
        model += q[(u, v, w)] >= -(len(vertex_P_ids) - 1) * y[(u, v, w)]

    # Spanning tree
    model += (
        pl.lpSum([y[(u, v, w)] for (u, v, w) in edge_P_ids]) == len(vertex_P_ids) - 1
    )

    # DC flow:
    # If lines are switched off, their corresponding beta needs to
    # be switched off as well. This is best modeled using an either-or
    # constraints.
    for e, ep in cross_edges:
        i, j = e[0], e[1]
        # Line is still activated
        model += fplus[e] - fmin[e] >= b[e] * (theta[i] - theta[j]) - (1 - y[ep]) * M[e]
        model += fplus[e] - fmin[e] <= b[e] * (theta[i] - theta[j]) + (1 - y[ep]) * M[e]

        # Line is deactivated
        model += f[e] >= -y[ep] * M[e]
        model += f[e] <= y[ep] * M[e]

    for i, j, k in internal_edges:
        model += fplus[(i, j, k)] - fmin[(i, j, k)] == b[(i, j, k)] * (
            theta[i] - theta[j]
        )

    # KCL
    for u in vertex_ids:
        model += (
            pl.lpSum(
                [di * (fplus[(i, j, k)] - fmin[(i, j, k)])]
                for (i, j, k), di in neighbors[u]
            )
            == p[u]
        )

    # Refbus
    model += theta[n - 1] == 0
    # breakpoint()
    if solver:
        model.solve(solver)
    else:
        model.solve()  # Uses the built-in solver

    # Gather the results
    new_flows = {
        G[i][j][k]["name"]: fplus[(i, j, k)].value() - fmin[(i, j, k)].value()
        for (i, j, k), v in f.items()
    }
    congestion = {
        G[i][j][k]["name"]: v.value() / c[(i, j, k)] for (i, j, k), v in f.items()
    }
    gamma = model.objective.value()
    removed_lines = [
        GP.get_edge_data(*e)["name"] for e, v in y.items() if v.value() == 0
    ]
    delta_f = {
        G.get_edge_data(*e)["name"]: abs(G.get_edge_data(*e)["weight"] - v.value())
        for e, v in f.items()
        if G.get_edge_data(*e)["name"] not in removed_lines
    }
    power_flow_disruption = sum(
        [
            G.get_edge_data(*e)["weight"]
            for e in G.edges
            if G.get_edge_data(*e)["name"] in removed_lines
        ]
    )
    if power_flow_disruption:
        delta_f_normalized = {k: v / power_flow_disruption for k, v in delta_f.items()}

    congested_lines = [name for name, cg in congestion.items() if cg > 1 + 0.001]

    results = {
        "model": model,
        "f": new_flows,
        "congestion": congestion,
        "gamma": gamma,
        "running_time": model.solutionTime,
        "removed_lines": removed_lines,
        "delta_f": delta_f,
        "power_flow_disruption": power_flow_disruption,
        "congested_lines": congested_lines,
    }
    if power_flow_disruption:
        results["delta_f_normalized"] = delta_f_normalized

    return results


def _milp_pyomo(P, objective="congestion", solver=""):
    """
    ...
    """
    case = Case()
    netdict = case.netdict

    # Define a model
    model = pyo.ConcreteModel(f"Line Switching Problem: Minimize {objective}")

    # Declare decision variables
    model.gamma = pyo.Var(domain=pyo.NonNegativeReals)
    model.fabs = pyo.Var(netdict["lines"], domain=pyo.NonNegativeReals)
    model.fp = pyo.Var(netdict["lines"], domain=pyo.NonNegativeReals)
    model.fm = pyo.Var(netdict["lines"], domain=pyo.NonNegativeReals)
    model.theta = pyo.Var(netdict["buses"], domain=pyo.Reals)
    # model.y = pyo.Var(rg["lines"], domain=pyo.Binary)
    # model.q = pyo.Var(rg["lines"], domain=pyo.Reals)

    # Declare objective value
    if objective == "congestion":
        model.objective = pyo.Objective(expr=model.gamma, sense=pyo.minimize)
    else:
        raise ValueError(f"Objective {objective} is not valid.")

    # Declare constraints
    model.max_congestion_constraint = pyo.Constraint(
        netdict["lines"],
        rule=lambda m, i, j: m.gamma >= m.fabs[i, j] / netdict["lines"][(i, j)]["c"],
    )

    model.outgoing_flow = pyo.Expression(
        netdict["buses"],
        rule=lambda m, i: sum(
            m.fp[i, j] - m.fm[i, j]
            for j in netdict["buses"]
            if (i, j) in netdict["lines"]
        ),
    )

    model.incoming_flow = pyo.Expression(
        netdict["buses"],
        rule=lambda m, i: sum(
            m.fp[j, i] - m.fm[j, i]
            for j in netdict["buses"]
            if (j, i) in netdict["lines"]
        ),
    )

    model.flow_conservation = pyo.Constraint(
        netdict["buses"],
        rule=lambda m, i: m.outgoing_flow[i] - m.incoming_flow[i]
        == netdict["buses"][i]["p_mw"],
    )

    model.susceptance = pyo.Constraint(
        netdict["lines"],
        rule=lambda m, i, j: m.fp[i, j] - m.fm[i, j]
        == netdict["lines"][(i, j)]["b"] * (m.theta[i] - m.theta[j]),
    )

    model.abs_flows = pyo.Constraint(
        netdict["lines"], rule=lambda m, *e: m.fabs[e] == m.fp[e] + m.fm[e]
    )
    model.flows_upper_bound = pyo.Constraint(
        netdict["lines"], rule=lambda m, *e: m.fabs[e] <= netdict["lines"][e]["c"]
    )

    # Solve
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    result = solver.solve(model)

    # Print solution
    print(
        f"**Solver status:** *{result.solver.status}, {result.solver.termination_condition}*"
    )
    # display(Markdown(f"**Minimizes objective value to:** ${model.objective():.4f}$"))

    # objective_value = model.objective()
    import ipdb

    ipdb.set_trace()

    return 0
