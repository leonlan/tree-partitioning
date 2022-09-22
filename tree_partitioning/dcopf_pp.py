import pandapower as pp

from tree_partitioning.classes import _nx_utils, _pp_utils


def dcopf_pp(G, net, switched_lines):
    """
    Solve DC optimal power flow for the passed-in graph G using pandapower.

    Return a new graph with adjusted power flows and the load shedding.
    """
    # Deactivate the lines from the pandapower net and run OPF
    names = [G.edges[line]["name"] for line in switched_lines]
    net.line.loc[net.line.name.isin(names), "in_service"] = False
    net.trafo.loc[net.trafo.name.isin(names), "in_service"] = False
    pp.rundcopp(net)

    # Create new graph
    netdict = _pp_utils._netdict_from_pp_net(net, merge_lines=False)
    return _nx_utils._G_from_netdict(netdict)
