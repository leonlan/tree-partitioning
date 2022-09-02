import networkx as nx
import pyomo.environ as pyo

from tree_partitioning.constants import _EPS


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

    if abs(power_imbalance) < _EPS:
        model.no_gen_adj = pyo.Constraint(expr=model.gen_adjustment == 1)
        model.no_load_shedding = pyo.Constraint(expr=model.load_shedding == 1)
    elif power_imbalance >= 0:
        model.no_gen_adj = pyo.Constraint(expr=model.gen_adjustment == 1)
    else:
        model.no_load_shedding = pyo.Constraint(expr=model.load_shedding == 1)

    # Declare objective value
    @model.Objective(sense=pyo.maximize)
    def objective(m):
        """
        Maximize the gen adjustment or load shedding needed to get to a
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
