import pyomo.environ as pyo

from ._base_line_switching import _base_line_switching


def maximum_congestion(G, partition, **kwargs):
    m = _base_line_switching(G, partition, **kwargs)

    m.gamma = pyo.Var(domain=pyo.NonNegativeReals)
    m.flow = pyo.Var(m.lines, domain=pyo.Reals)
    m.theta = pyo.Var(m.buses, domain=pyo.Reals)

    # TODO why does big M need to this big?
    m.M = pyo.Param(
        m.lines,
        initialize={line: data["c"] * 20 for line, data in m.line_data.items()},
        within=pyo.Reals,
    )

    @m.Objective(sense=pyo.minimize)
    def objective(m):
        return m.gamma

    @m.Constraint(m.lines)
    def congestion_1(m, *line):
        return m.gamma >= m.flow[line] / m.line_data[line]["c"]

    @m.Constraint(m.lines)
    def congestion_2(m, *line):
        return m.gamma >= -m.flow[line] / m.line_data[line]["c"]

    @m.Expression(m.lines)
    def dc_flow(m, *line):
        i, j, _ = line
        return m.line_data[line]["b"] * (m.theta[i] - m.theta[j])

    @m.Constraint(m.internal)
    def internal_edges_flow(m, *line):
        return m.flow[line] == m.dc_flow[line]

    @m.Constraint(m.buses)
    def flow_conservation(m, bus):
        out = sum(m.flow[i, j, idx] for (i, j, idx) in m.lines if i == bus)
        inc = sum(m.flow[i, j, idx] for (i, j, idx) in m.lines if j == bus)
        return out - inc == m.bus_data[bus]["p_mw"]

    # For some reason, cross_edges elements are flattened tuples
    # (u, v, i, j, k) instead of (u, v, (i, j, k))
    @m.Constraint(m.cross)
    def active_cross_edge_1(m, *ce):
        line = tuple(ce[2:])
        rhs = m.dc_flow[line] - (1 - m.active_line[line]) * m.M[line]
        return m.flow[line] >= rhs

    @m.Constraint(m.cross)
    def active_cross_edge_2(m, *ce):
        line = tuple(ce[2:])
        rhs = m.dc_flow[line] + (1 - m.active_line[line]) * m.M[line]
        return m.flow[line] <= rhs

    @m.Constraint(m.cross)
    def inactive_cross_edge_1(m, *ce):
        line = tuple(ce[2:])
        return m.flow[line] >= -m.active_line[line] * m.M[line]

    @m.Constraint(m.cross)
    def inactive_cross_edge_2(m, *ce):
        line = tuple(ce[2:])
        return m.flow[line] <= m.active_line[line] * m.M[line]

    return m
