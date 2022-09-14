import pyomo.environ as pyo


def base_model(case, generators):
    """
    Base model for single-stage tree partitioning formulations.
    """
    # Temporary vars to make life easy
    _buses, _lines = case.G.nodes, case.G.edges
    _clusters = generators

    # Define a model
    m = pyo.ConcreteModel()

    # Defines set indices
    m.buses = pyo.Set(initialize=_buses.keys())
    m.lines = pyo.Set(initialize=_lines.keys())
    m.clusters = pyo.Set(initialize=_clusters.keys())

    # Define parameters data sets
    m.bus_data = pyo.Param(m.buses, initialize=_buses, within=pyo.Any)
    m.line_data = pyo.Param(m.lines, initialize=_lines, within=pyo.Any)
    m.cluster_data = pyo.Param(m.clusters, initialize=_clusters, within=pyo.Any)

    # TODO why does big M need to this big?
    m.M = pyo.Param(
        m.lines,
        initialize={line: data["c"] * 10 for line, data in m.line_data.items()},
        within=pyo.Reals,
    )

    # Declare decision variables
    m.assign_bus = pyo.Var(m.buses, m.clusters, domain=pyo.Binary)
    m.assign_line = pyo.Var(m.lines, m.clusters, domain=pyo.Binary)
    m.active_cross_edge = pyo.Var(m.lines, domain=pyo.Binary)
    m.active_line = pyo.Var(m.lines, domain=pyo.Binary)
    m.commodity_flow = pyo.Var(m.lines, domain=pyo.Reals)

    # Declare constraints
    @m.Constraint(m.buses)
    def assign_generator_to_corresponding_cluster(m, bus):
        for cluster, nodes in m.cluster_data.items():
            if bus in nodes:
                return m.assign_bus[bus, cluster] == 1

        return pyo.Constraint.Skip

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

    @m.Constraint(m.lines)
    def relate_assignment_to_activeness(m, *line):
        rhs1 = sum(m.assign_line[line, cluster] for cluster in m.clusters)
        rhs2 = m.active_cross_edge[line]
        lhs = m.active_line[line]
        return rhs1 + rhs2 == lhs

    @m.Constraint()
    def exactly_k_minus_1_cross_edges(m):
        return sum(m.active_cross_edge[line] for line in m.lines) == len(m.clusters) - 1

    @m.Constraint()
    def commodity_flow_source_bus(m):
        rhs1 = sum([m.commodity_flow[i, j, idx] for (i, j, idx) in m.lines if i == 0])
        rhs2 = sum([m.commodity_flow[j, i, idx] for (j, i, idx) in m.lines if i == 0])
        return rhs1 - rhs2 == len(m.buses) - 1

    @m.Constraint(m.buses)
    def commodity_flow_sink_buses(m, bus):
        if bus != 0:
            rhs1 = sum([m.commodity_flow[i, j, x] for (i, j, x) in m.lines if i == bus])
            rhs2 = sum([m.commodity_flow[j, i, x] for (j, i, x) in m.lines if i == bus])
            return rhs1 - rhs2 == -1

        return pyo.Constraint.Skip

    @m.Constraint(m.lines)
    def commodity_flow_only_if_active_lhs(m, *line):
        return -(len(m.buses) - 1) * m.active_line[line] <= m.commodity_flow[line]

    @m.Constraint(m.lines)
    def commodity_flow_only_if_active_rhs(m, *line):
        return m.commodity_flow[line] <= (len(m.buses) - 1) * m.active_line[line]

    return m
