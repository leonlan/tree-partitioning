import pyomo.environ as pyo

from .base_model import base_model


def power_imbalance(G, generators, **kwargs):
    model = base(G, generators, **kwargs)

    model.power_imbalance = pyo.Var(m.clusters, domain=pyo.NonNegativeReals)

    @model.Expression(m.clusters)
    def total_power_imbalance(m, cluster):
        return sum(
            data["p_mw"] * m.assign_bus[bus, cluster]
            for bus, data in m.bus_data.items()
        )

    @model.Constraint(m.clusters)
    def lower_bound_power_imbalance(m, cluster):
        return m.power_imbalance[cluster] >= m.total_power_imbalance[cluster]

    @model.Constraint(m.clusters)
    def upper_bound_power_imbalance(m, cluster):
        return -m.power_imbalance[cluster] <= m.total_power_imbalance[cluster]

    @model.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(m.power_imbalance[r] for r in m.clusters)

    return model
