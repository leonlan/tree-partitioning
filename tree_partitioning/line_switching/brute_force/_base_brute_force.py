from itertools import combinations
from time import perf_counter
from typing import List

import networkx as nx

from tree_partitioning.classes import Partition
from tree_partitioning.utils import compute_cross_edges


def _base_brute_force(G, partition: Partition, objective_function, **kwargs):
    """
    Solve the Optimal Line Switching problem using brute force enumeration.

    Given a graph G and a partition P, the brute force approach evaluates
    all possible line switching actions that transform P into a tree partition
    of the post switching graph G_post.

    An objective function needs to be provided to evaluate the impact of
    the switching actions. This function should take as input the graph G
    and the set of line switching actions E and returns the cost.
    """
    cross_edges = compute_cross_edges(G, partition)  # (u, v, (i, j, idx))

    # Compute all possible combinations of cross edges that form a spanning tree
    # and store the corresponding candidate switching actions
    candidates = []
    for subset in combinations(cross_edges, r=len(partition) - 1):
        if _is_tree_graph(subset, partition.clusters.keys()):
            switching_lines = [edge[2] for edge in cross_edges if edge not in subset]
            candidates.append(switching_lines)

    best_cost, best_lines = None, None
    for switched_lines in candidates:
        cost = objective_function(G, switched_lines)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_lines = switched_lines

    return best_cost, best_lines


def _is_tree_graph(cross_edges, clusters):
    """
    Checks whether the edges form a tree graph. Using the passed-in
    cross edges and clusters, we form a new graph and check whether the
    edges form a tree on it.

    Cross edges are given as a list of tuples (u, v, (i, j, idx)),
    where u, v are the cluster indices and (i, j, idx) are the line indices.
    """
    graph = nx.Graph()
    graph.add_nodes_from(clusters)
    graph.add_edges_from([e[:2] for e in cross_edges])
    return nx.is_tree(graph)
