import numpy as np
import pyomo.environ as pyo

from tree_partitioning.classes import Partition, ReducedGraph, Solution
from tree_partitioning.constants import _EPS


def _base_line_switching(G, partition: Partition, objective="congestion", **kwargs):
    """
    Base model for optimal line switching models. The following properties should hold:
    - The reduced graph is connected; and
    - The reduced graph is a tree, i.e., exactly k - 1 cross edges are active.
    """
    _buses, _lines = G.nodes, G.edges

    m = pyo.ConcreteModel()

    m.buses = pyo.Set(initialize=_buses.keys())
    m.lines = pyo.Set(initialize=_lines.keys())
    m.clusters = pyo.Set(initialize=partition.clusters.keys())

    m.bus_data = pyo.Param(m.buses, initialize=_buses, within=pyo.Any)
    m.line_data = pyo.Param(m.lines, initialize=_lines, within=pyo.Any)

    # TODO refactor this
    get_ep = partition.membership
    _endpoints = {(i, j, _): (get_ep[i], get_ep[j]) for (i, j, _) in m.lines}

    m.endpoints = pyo.Param(m.lines, initialize=_endpoints, within=pyo.Any)
    m.is_cross = pyo.Param(
        m.lines,
        initialize={line: u != v for line, (u, v) in m.endpoints.items()},
        within=pyo.Boolean,
    )

    m.cross = pyo.Set(
        initialize=[(*m.endpoints[line], line) for line in m.lines if m.is_cross[line]]
    )
    m.internal = pyo.Set(initialize=[line for line in m.lines if not m.is_cross[line]])

    m.active_line = pyo.Var(m.lines, domain=pyo.Binary)
    m.com_flow = pyo.Var(m.clusters, m.clusters, m.lines, domain=pyo.Reals)

    @m.Constraint(m.cross)
    def commodity_flow_upper_bound(m, *ce):
        line = ce[2:]
        return m.com_flow[ce] <= (len(m.clusters) - 1) * m.active_line[line]

    @m.Constraint(m.cross)
    def commodity_flow_lower_bound(m, *ce):
        line = ce[2:]
        return -(len(m.clusters) - 1) * m.active_line[line] <= m.com_flow[ce]

    @m.Constraint(m.clusters)
    def commodity_flow(m, cluster):
        out = sum([m.com_flow[ce] for ce in m.cross if ce[0] == cluster])
        inc = sum([m.com_flow[ce] for ce in m.cross if ce[1] == cluster])

        quantity = len(m.clusters) - 1 if cluster == 0 else -1
        return out - inc == quantity

    @m.Constraint()
    def exactly_k_min_1_cross_edges(m):
        lhs = sum([m.active_line[tuple(line)] for (u, v, *line) in m.cross])
        return lhs == len(m.clusters) - 1

    return m
