import pyomo.environ as pyo

from ._base_model import _base_model


def recursive_model(case, generators):
    m = _base_model(case, generators)
    m.groups = pyo.Set(initialize=generators.keys())
    m.group_data = pyo.Param(m.groups, initialize=generators, within=pyo.Any)

    m.assign_generator_to_corresponding_cluster.deactivate()

    m.assign_group = pyo.Var(generators, m.clusters, domain=pyo.Binary)

    @m.Constraint(m.clusters)
    def at_least_one_group_per_cluster(m, cluster):
        return sum(m.assign_group[group, cluster] for group in m.groups) >= 1

    @m.Constraint(m.groups)
    def exactly_one_cluster_per_group(m, group):
        return sum(m.assign_group[group, cluster] for cluster in m.clusters) == 1

    @m.Constraint(m.groups, m.clusters)
    def generators_group_within_same_cluster(m, group, cluster):
        generators = m.group_data[group]
        return sum(m.assign_bus[gen, cluster] for gen in generators) == m.assign_group[
            group, cluster
        ] * len(generators)

    return m
