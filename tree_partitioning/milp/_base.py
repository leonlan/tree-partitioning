import pyomo.environ as pyo


def _base(G, n_clusters, **kwargs) -> pyo.ConcreteModel:
    """
    Base model used for partitioning and tree partitioning models.
    """
    _buses, _lines = G.nodes, G.edges

    # Define a model
    m = pyo.ConcreteModel()

    # Defines set indices
    m.buses = pyo.Set(initialize=_buses.keys())
    m.lines = pyo.Set(initialize=_lines.keys())
    m.clusters = pyo.Set(initialize=range(n_clusters))

    # Define parameters data sets
    m.bus_data = pyo.Param(m.buses, initialize=_buses, within=pyo.Any)
    m.line_data = pyo.Param(m.lines, initialize=_lines, within=pyo.Any)

    # Declare decision variables
    m.assign_bus = pyo.Var(m.buses, m.clusters, domain=pyo.Binary)
    m.assign_line = pyo.Var(m.lines, m.clusters, domain=pyo.Binary)
    m.active_line = pyo.Var(m.lines, domain=pyo.Binary)

    @m.Constraint(m.buses)
    def assign_bus_to_exactly_one_cluster(m, bus):
        return sum(m.assign_bus[bus, cluster] for cluster in m.clusters) == 1

    @m.Constraint(m.clusters, m.lines)
    def assign_line_only_if_assign_bus_i(m, cluster, *line):
        i, j, _ = line
        return m.assign_line[line, cluster] <= m.assign_bus[i, cluster]

    @m.Constraint(m.clusters, m.lines)
    def assign_line_only_if_assign_bus_j(m, cluster, *line):
        i, j, _ = line
        return m.assign_line[line, cluster] <= m.assign_bus[j, cluster]

    @m.Constraint(m.clusters, m.lines)
    def assign_line_if_both_assign_buses(m, cluster, *line):
        i, j, _ = line
        lhs = m.assign_line[line, cluster]
        rhs = m.assign_bus[i, cluster] + m.assign_bus[j, cluster] - 1
        return lhs >= rhs

    return m
