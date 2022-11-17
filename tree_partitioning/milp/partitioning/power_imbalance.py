import pyomo.environ as pyo

from ._base_partitioning import _base_partitioning


def power_imbalance(G, generators, recursive=False, **kwargs):
    model = _base_partitioning(G, generators, recursive, **kwargs)

    model.power_imbalance = pyo.Var(model.clusters, domain=pyo.NonNegativeReals)

    @model.Expression(model.clusters)
    def total_power_imbalance(m, cluster):
        return sum(
            data["p_load_total"] - data["p_gen_total"] * model.assign_bus[bus, cluster]
            for bus, data in model.bus_data.items()
        )

    @model.Constraint(model.clusters)
    def lower_bound_power_imbalance(m, cluster):
        return model.power_imbalance[cluster] >= model.total_power_imbalance[cluster]

    @model.Constraint(model.clusters)
    def upper_bound_power_imbalance(m, cluster):
        return -model.power_imbalance[cluster] <= model.total_power_imbalance[cluster]

    @model.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(model.power_imbalance[r] for r in model.clusters)

    return model
