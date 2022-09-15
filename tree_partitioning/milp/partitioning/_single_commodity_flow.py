import pyomo.environ as pyo


def _single_commodity_flow(m):
    """
    Single commodity flow constraints to ensure connectivity within each cluster.
    These constraints do not assume that there are reference buses that can be
    used as source bus.
    """
    m.source_bus = pyo.Var(m.buses, m.clusters, domain=pyo.Binary)
    m.commodity = pyo.Var(m.lines, domain=pyo.Reals)
    m.M = len(m.buses) - len(m.clusters)

    @m.Constraint(m.lines)
    def relate_assignment_to_activeness(m, *line):
        lhs = sum(m.assign_line[line, cluster] for cluster in m.clusters)
        rhs = m.active_line[line]

        return lhs == rhs

    @m.Constraint(m.clusters)
    def one_source_bus_per_cluster(m, cluster):
        return sum(m.source_bus[bus, cluster] for bus in m.buses) == 1

    @m.Constraint(m.buses, m.clusters)
    def source_bus_if_assigned_bus(m, bus, cluster):
        return m.source_bus[bus, cluster] <= m.assign_bus[bus, cluster]

    @m.Constraint(m.lines)
    def commodity_only_if_active_lhs(m, *line):
        return -m.M * m.active_line[line] <= m.commodity[line]

    @m.Constraint(m.lines)
    def commodity_only_if_active_rhs(m, *line):
        return m.commodity[line] <= m.M * m.active_line[line]

    @m.Constraint(m.buses)
    def commodity_flow_balance(m, bus):
        out = [m.commodity[i, j, idx] for (i, j, idx) in m.lines if i == bus]
        inc = [m.commodity[j, i, idx] for (j, i, idx) in m.lines if i == bus]
        is_source = sum(m.source_bus[bus, cluster] for cluster in m.clusters)

        return sum(out) - sum(inc) >= 1 - m.M * is_source
