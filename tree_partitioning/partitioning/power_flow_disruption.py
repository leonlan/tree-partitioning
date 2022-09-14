import pyomo.environ as pyo

from .base_model import base_model
from .recursive_model import recursive_model


def power_flow_disruption(G, generators, **kwargs):
    if kwargs["recursive"]:
        model = recursive_model(G, generators, **kwargs)
    else:
        model = base_model(G, generators, **kwargs)

    @model.Objective(sense=pyo.minimize)
    def objective(m):
        return sum(
            data["f"] * (1 - m.active_line[line]) for line, data in m.line_data.items()
        )

    return model
