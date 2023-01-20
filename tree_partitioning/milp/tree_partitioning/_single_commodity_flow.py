import pyomo.environ as pyo


def _single_commodity_flow(m):
    """
    Single commodity flow constraints to ensure graph connectivity.
    Assumes the bus with zero index is the source bus.
    """
    m.commodity_flow = pyo.Var(m.lines, domain=pyo.Reals)

    source_bus = min(m.buses)

    @m.Constraint()
    def commodity_flow_source_bus(m):
        rhs1 = sum(
            [m.commodity_flow[i, j, idx] for (i, j, idx) in m.lines if i == source_bus]
        )
        rhs2 = sum(
            [m.commodity_flow[j, i, idx] for (j, i, idx) in m.lines if i == source_bus]
        )
        return rhs1 - rhs2 == len(m.buses) - 1

    @m.Constraint(m.buses)
    def commodity_flow_sink_buses(m, bus):
        if bus != source_bus:
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
