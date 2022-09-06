import networkx as nx
import pyomo.environ as pyo

from tree_partitioning.constants import _EPS


def dcopf(G):
    """
    Solve DC power flow for the passed-in graph G with possibly load shedding
    or generation adjustments.

    Return a new graph with adjusted power flows and the load shedding.
    """
    buses, lines = G.nodes, G.edges

    # Define a model
    model = pyo.ConcreteModel("DC-PF with adjustments")

    # Define variables
    model.adj = pyo.Var(buses, domain=pyo.NonNegativeReals)
    model.adjm = pyo.Var(buses, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.adjp = pyo.Var(buses, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.theta = pyo.Var(buses, domain=pyo.Reals)
    model.flow = pyo.Var(lines, domain=pyo.Reals)

    # Either generation loss or load shedding
    # Positive power imbalance indicates more load than generation
    power_imbalance = sum(nx.get_node_attributes(G, "p_mw").values())

    # Declare objective value
    @model.Objective(sense=pyo.minimize)
    def objective(m):
        """
        Maximize the gen adjustment or load shedding needed to get to a
        convergent solution.
        """
        return sum(m.adj[bus] * abs(buses[bus]["p_mw"]) for bus in buses)

    @model.Constraint(buses)
    def abs_adjustments(m, bus):
        return m.adj[bus] == m.adjp[bus] + m.adjm[bus]

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
        rhs = (1 + m.adjp[bus] - m.adjm[bus]) * buses[bus]["p_mw"]

        return lhs == rhs

    @model.Constraint(lines)
    def susceptance(m, *line):
        i, j, _ = line
        return m.flow[line] == lines[line]["b"] * (m.theta[i] - m.theta[j])

    @model.Constraint(lines)
    def no_congestion0(m, *line):
        return m.flow[line] <= lines[line]["c"]

    @model.Constraint(lines)
    def no_congestion1(m, *line):
        return m.flow[line] >= -lines[line]["c"]

    # Solve
    solver = pyo.SolverFactory("gurobi", solver_io="python")
    result = solver.solve(model, tee=False, options={"TimeLimit": 300})

    # # Print solution
    # print(f"{model.gen_adjustment.value=:.2f}, {model.load_shedding.value=:.2f}")

    # Make a new copy of the graph and change the power injections and flows
    H = G.copy()

    new_power_injections = {
        bus: (1 + model.adjp[bus].value - model.adjm[bus].value) * p_mw
        for bus, p_mw in nx.get_node_attributes(H, "p_mw").items()
    }

    print("Total adjusted: ", model.objective())

    nx.set_node_attributes(H, new_power_injections, "p_mw")

    new_flows = {k: v.value for k, v in model.flow.items()}
    nx.set_edge_attributes(H, new_flows, "f")

    return H, (power_imbalance if power_imbalance > 0 else 0)
