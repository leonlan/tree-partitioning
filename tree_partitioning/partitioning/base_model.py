import pyomo.environ as pyo

from ..milp import partitioning_model


def base_model(G, groups, **kwargs) -> pyo.ConcreteModel:
    """
    Solve the Tree Partition Identification minimizing transient stability.
    """
    _groups = groups

    m = partitioning_model(G, len(groups))
    m.clusters = pyo.Set(initialize=range(2))
    m.groups = pyo.Set(initialize=_groups.keys())

    m.group_data = pyo.Param(m.groups, initialize=_groups, within=pyo.Any)

    # Declare decision variables
    m.commodity = pyo.Var(m.lines, domain=pyo.Reals)

    # Declare constraints
    @m.Constraint(m.buses)
    def assign_generator_to_corresponding_cluster(m, bus):
        for group, gens in m.group_data.items():
            if bus in gens:
                return m.assign_bus[bus, group] == 1

        return pyo.Constraint.Skip

    @m.Constraint(m.lines)
    def relate_assignment_to_activeness(m, *line):
        lhs = sum(m.assign_line[line, cluster] for cluster in m.clusters)
        rhs = m.active_line[line]

        return lhs == rhs

    @m.Constraint(m.clusters)
    def commodity_source_bus(m, cluster):
        # Take the first generator in the generator group
        source = m.cluster_data[cluster][0]

        out = [m.commodity[i, j, idx] for (i, j, idx) in m.lines if i == source]
        inc = [m.commodity[j, i, idx] for (j, i, idx) in m.lines if i == source]

        return sum(out) - sum(inc) == m.cluster_size[cluster] - 1

    @m.Constraint(m.buses)
    def commodity_sink_buses(m, bus):
        # Skip all source generator nodes
        if bus in [gens[0] for _, gens in m.cluster_data.items()]:
            return pyo.Constraint.Skip

        rhs1 = sum([m.commodity[i, j, idx] for (i, j, idx) in m.lines if i == bus])
        rhs2 = sum([m.commodity[j, i, idx] for (j, i, idx) in m.lines if i == bus])

        return rhs1 - rhs2 == -1

    @m.Constraint(m.lines)
    def commodity_only_if_active_lhs(m, *line):
        return -(len(m.buses) - 1) * m.active_line[line] <= m.commodity[line]

    @m.Constraint(m.lines)
    def commodity_only_if_active_rhs(m, *line):
        return m.commodity[line] <= (len(m.buses) - 1) * m.active_line[line]

    return m
