import pyomo.environ as pyo


def _grouping_constraint(m, groups, recursive=False):
    _groups = groups
    m.groups = pyo.Set(initialize=_groups.keys())
    m.group_data = pyo.Param(m.groups, initialize=_groups, within=pyo.Any)

    if not recursive:

        @m.Constraint(m.buses)
        def assign_generator_to_corresponding_cluster(m, bus):
            for group, gens in m.group_data.items():
                if bus in gens:
                    return m.assign_bus[bus, group] == 1

            return pyo.Constraint.Skip

    if recursive:
        m.assign_group = pyo.Var(m.groups, m.clusters, domain=pyo.Binary)

        @m.Constraint()
        def fix_first_group(m):
            """
            The first group-cluster assignment can always be fixed.
            """
            group = tuple(m.groups)[0]
            cluster = tuple(m.clusters)[0]
            return m.assign_group[group, cluster] == 1

        @m.Constraint(m.clusters)
        def at_least_one_group_per_cluster(m, cluster):
            return sum(m.assign_group[group, cluster] for group in m.groups) >= 1

        @m.Constraint(m.groups)
        def exactly_one_cluster_per_group(m, group):
            return sum(m.assign_group[group, cluster] for cluster in m.clusters) == 1

        @m.Constraint(m.groups, m.clusters, m.buses)
        def generators_group_within_same_cluster(m, group, cluster, bus):
            if bus in m.group_data[group]:
                return m.assign_group[group, cluster] == m.assign_bus[bus, cluster]

            return pyo.Constraint.Skip
