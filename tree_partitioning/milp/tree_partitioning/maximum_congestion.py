import pyomo.environ as pyo

from tree_partitioning.constants import _EPS

from ._base_tree_partitioning import _base_tree_partitioning


def maximum_congestion(G, generators, **kwargs):
    """
    Solve the tree partitioning problem minimizing maximum congestion.
    """
    m = _base_tree_partitioning(G, generators, **kwargs)

    m.gamma = pyo.Var(domain=pyo.NonNegativeReals)
    m.flow = pyo.Var(m.lines, domain=pyo.Reals)
    m.theta = pyo.Var(m.buses, domain=pyo.Reals)

    # TODO why does big M need to this big?
    m.M = pyo.Param(
        m.lines,
        initialize={line: data["c"] * 10 for line, data in m.line_data.items()},
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

    # DC power flow constraints
    @m.Expression(m.buses)
    def outgoing_flow(m, bus):
        return sum(m.flow[line] for line in m.lines if line[0] == bus)

    @m.Expression(m.buses)
    def incoming_flow(m, bus):
        return sum(m.flow[line] for line in m.lines if line[1] == bus)

    @m.Constraint(m.buses)
    def flow_conservation(m, bus):
        return m.outgoing_flow[bus] - m.incoming_flow[bus] == m.bus_data[bus]["p_mw"]

    @m.Expression(m.lines)
    def dc_eq(m, *line):
        i, j, _ = line
        return m.line_data[line]["b"] * (m.theta[i] - m.theta[j])

    @m.Constraint(m.lines)
    def flow_active_line_1(m, *line):
        return (
            m.flow[line] <= m.dc_eq[line] + m.M[line] * (1 - m.active_line[line]) + _EPS
        )

    @m.Constraint(m.lines)
    def flow_active_line_2(m, *line):
        return (
            m.flow[line] >= m.dc_eq[line] - m.M[line] * (1 - m.active_line[line]) - _EPS
        )

    @m.Constraint(m.lines)
    def flow_inactive_line_1(m, *line):
        return m.flow[line] <= m.M[line] * m.active_line[line]

    @m.Constraint(m.lines)
    def flow_inactive_line_2(m, *line):
        return m.flow[line] >= -m.M[line] * m.active_line[line]

    return m
