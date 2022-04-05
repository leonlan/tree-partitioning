from itertools import combinations

import networkx as nx
import pandapower as pp

from tree_partitioning.classes import Case, Partition, Solution, ReducedGraph


def brute_force(partition: Partition) -> Solution:
    """
    Solve the Line Switching Problem using brute force enumeration.
    """
    case = Case()
    net = case.net
    G = case.G
    reduced_graph = ReducedGraph(G, partition)

    # Compute all possible combinations of cross edges that could be a spanning tree
    # Then store all the cross edges that are not part of the selected ones
    candidates = []
    for edges in combinations(reduced_graph.cross_edges, r=len(partition) - 1):
        if _is_tree_graph(edges):
            switching_lines = [
                edge[2] for edge in reduced_graph.cross_edges if edge not in edges
            ]
            candidates.append(switching_lines)

    # For each of the candidate line sets, deactivate the lines in pandapower
    # and rerun power flows to obtain congestion
    best_lines, best_gamma = None, 100

    for lines in candidates:
        post_switching_net = _deactivate_lines_pp(net, lines)
        pp.rundcpp(post_switching_net)
        new_gamma = _max_loading_percent(post_switching_net)

        if new_gamma < best_gamma:
            best_lines = lines
            best_gamma = new_gamma

    return Solution(partition, best_lines)


def _is_tree_graph(edges: list):
    """
    Checks whether the edges form a tree graph.

    Edges should be given as a list of tuples (u, v, ...),
    where u and v indicate the node index. All other arguments
    are ignored.
    """
    graph = nx.Graph()
    graph.add_edges_from([e[:2] for e in edges])
    return nx.is_tree(graph)


def _deactivate_lines_pp(net, lines):
    """Deactivate lines of a pandapower network. """
    # Get the line names first from the lines
    netdict = Case().netdict
    line_names = [netdict["lines"][line]["name"] for line in lines]

    net = pp.copy.deepcopy(net)
    net.line.loc[net.line["name"].isin(line_names), "in_service"] = False
    net.trafo.loc[net.trafo["name"].isin(line_names), "in_service"] = False

    return net


def _max_loading_percent(net):
    """Compute the maximum loading percent of net."""
    gl = max(net.res_line.loading_percent) / 100

    # Some cases do not have transformers
    try:
        gt = max(net.res_trafo.loading_percent) / 100
    except ValueError:
        return gl
    return max(gl, gt)
