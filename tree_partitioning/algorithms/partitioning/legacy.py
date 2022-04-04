"""
Legacy functions used for the Optimal Bridge-blocks Identification (OBI).
"""
import copy
from collections import defaultdict
from itertools import product
import time as time

import igraph as ig
import pandas as pd
import networkx as nx
import pandapower as pp
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

from tree_partitioning.classes import Partition

from .bridge_block_refinement import networkanalysis, spectralclustering


def compute_rg_sg(C, n_clusters=4, goal="max", methods=["FastGreedy"]):
    """Bridge block refinement procedure using the one shot algorithm.

    The one shot algorithm solves the OBI-k and OSP seperately, where the OSP
    problem is solved using total enumeration.

    Args:
    - C (Case): Case object representing the grid
    - n_clusters: Number of target bridge blocks
    - goal: Optimization goal (default: max)
    - methods: List of methods for spectral clustering (default: all)

    Returns:
    - ...
    """
    net = pp.copy.deepcopy(C.net)
    igg = C.igg.copy()
    # Obtain the initial BBD
    block_partition, bridgeidx = networkanalysis(igg, verboseflag=False)

    # Statistics to keep track of
    results = pd.DataFrame(
        columns=[
            "runtime",
            "modularity",
            "max_cong",
            "avg_cong",
            "#congested_lines",
            "cross_edges",
            "spanning_trees",
            "#active_lines",
            "#removed_lines",
            "percentage_removed_lines",
            "lbbsize",
            "bcount",
            "bsizes_split",
            "block_sizes_post",
            "best_scenario",
            "removed_lines_type",
        ],
        index=methods,
    )

    # For each method, compute the results from the BBR
    # TODO: Change i varname to more descriptive (e.g. method)
    # TODO: Changed it, but check if this works.
    for method in results.index:
        t = time.time()
        G = igg.copy()
        activelines = G.ecount()

        # Find current bridge-blocks partition and bridges  of the graph
        block_partition, bridgeidx = networkanalysis(G)

        # Fastgreedy can't deal with multiple edges, which are then merged in a local copy
        if method == "FastGreedy":
            G.simplify(combine_edges=dict(weight="sum"))

        # Find pre-existing bridges and bridge-blocks of the graph and the biggest one in size
        sg = G.subgraph(max(list(block_partition), key=len))
        lbbsize = max(map(len, list(block_partition)))

        # The induced subgraph corresponding to the biggest bridge-block gets split into n_clusters
        if method == "FastGreedy":
            sgpartition = sg.community_fastgreedy(weights="weight").as_clustering(
                n=n_clusters
            )
        else:
            sgpartition = spectralclustering(
                sg, n_clusters, matrix=method, weight="weight"
            )

        # Cluster sizes and modularity score of the selected partition of sg
        bsplit_pre = list(map(len, list(sgpartition)))
        Q = sg.modularity(sgpartition, weights="weight")

        # The original graph partition gets updated accordingly
        # TODO: Understand this part. In what sense to the partitions gets updated?
        newm = block_partition.membership  # new membership vector for the full graph
        sgcommunities = []
        sgpm = np.array(sgpartition.membership)
        for c in range(n_clusters):
            if (
                c == 0
            ):  # For the first cluster in sg, leave the membership unchanged in newm and store the value in sg.communities
                idxcommunity = np.where(sgpm == c)[0]
                sgcommunities.append(
                    newm[G.vs.select(name_eq=sg.vs["name"][idxcommunity[0]])[0].index]
                )
            else:
                idxcommunity = np.where(sgpm == c)[
                    0
                ]  # find indices of nodes that belong to community c
                newcommunityindex = (
                    max(newm) + 1
                )  # community c inside subgraph gets updated index, which is equal to the current max index +1
                sgcommunities.append(newcommunityindex)
                for idx in [sg.vs["name"][sgidx] for sgidx in idxcommunity]:
                    newm[
                        G.vs.select(name_eq=idx)[0].index
                    ] = newcommunityindex  # all nodes that belong to community c gets updated memberhip in the full graph partition
        G.vs["community"] = newm
        npartition = ig.VertexClustering(
            G, membership=newm
        )  # new partition for the full graph
        bsize_pre = [npartition.subgraph(i).vcount() for i in range(len(npartition))]

        # The following for loop deals with the selected partition possibly having disconnected components
        connectedpartition_flag = True
        for ip in range(len(npartition)):
            if not (npartition.subgraph(ip).is_connected()):
                connectedpartition_flag = False
                for k in range(1, len(npartition.subgraph(ip).clusters())):
                    newcommunityindex = max(newm) + 1
                    sgcommunities.append(newcommunityindex)
                    v = [
                        npartition.subgraph(ip).vs["name"][j]
                        for j in npartition.subgraph(ip).clusters()[k]
                    ]
                    for idx in v:
                        newm[G.vs.select(name_eq=idx)[0].index] = newcommunityindex
                        G.vs.select(name_eq=idx)[0]["community"] = newcommunityindex
        if not (connectedpartition_flag):
            #             print('Flag: partition with disconnected clusters')
            G.vs["community"] = newm
            npartition = ig.VertexClustering(G, membership=newm)

        # final partition is casted into a partition of the original graph igg
        npartition = ig.VertexClustering(igg, membership=newm)
        bsize_post = [npartition.subgraph(i).vcount() for i in range(len(npartition))]

        # color attribute is given to the graph nodes depending on the cluster they belong to
        igg.vs["community"] = newm
        igg.vs["color"] = ig.drawing.colors.ClusterColoringPalette(
            len(npartition)
        ).get_many(npartition.membership)

        # reduced graph obtained by first collapsing vertex in the same cluster and then removing loops
        # multiple edges are purposefully left
        # each bridge-block has a 'name' attribute equal to the lowest vertex 'name' that it contains
        # each bridge-block has the color of the nodes in the corresponding cluster
        reducedgraph = igg.copy()
        reducedgraph.contract_vertices(npartition.membership, combine_attrs=list)
        reducedgraph.simplify(multiple=False, loops=True)
        # ig.plot(reducedgraph, vertex_label=reducedgraph.vs["community"])

        # Focuses on the subgraph of the reduced graph where the pruning is needed
        rg_sg = reducedgraph.subgraph(sgcommunities)
    return rg_sg


def make_reduced_subgraph(igg, partitioning):
    """Returns the reduced subgraph based on the partitioning.

    Args:
    - igg (igraph): igraph representing the network
    - partitioning (dict): Node indices assigned with partition number

    Returns:
    - rg_sg (igraph): Reduced subgraph of the network
    """
    rg_sg = igg.copy()
    L = [partitioning[i] for i in range(igg.vcount())]  # Membership list
    npartition = ig.VertexClustering(igg, membership=L)
    rg_sg.contract_vertices(npartition.membership, combine_attrs=list)
    rg_sg.simplify(multiple=False, loops=True)

    return rg_sg


class _Partition(dict):
    """Vertex partition class.

    Standard representation is a dictionary with clusters as keys and
    a list of vertex ids as values.
    """

    def __init__(self, *arg, **kw):
        super(_Partition, self).__init__(*arg, **kw)
        self.size = len(self.keys())

    @classmethod
    def from_list(cls, L):
        """Construct from a list."""
        D = defaultdict(list)
        for i, l in enumerate(L):
            D[l].append(i)

        return cls(D)

    def vertex2cluster(self):
        """Index the Partition by vertex number."""
        val2key = dict()
        for k, vals in self.items():
            for val in vals:
                val2key[val] = k
        return val2key


def extend_partition(G, P):
    """Extends a partition to include the entire graph.

    In some function, the partitions are computed only on the largest non-trivial
    bridge-block. This leaves out the smaller bridge-blocks, but we need those
    for initialization in the MILP.

    The idea is to run a shortest path from each excluded node. The first node
    it reaches gives the corresponding cluster number it belongs to. Note that
    there exists a unique node with minimum length due to the existence of a
    bridge.
    """
    # G is a MultiDiGraph, but we are only interested in undirected shortest paths
    H = G.to_undirected()

    dist = dict(nx.all_pairs_shortest_path_length(H))
    Q = defaultdict(list)
    v2c = P.vertex2cluster()
    V = P.vertex2cluster().keys()

    for u in G.nodes:
        if u not in V:
            min_dist = 10000
            min_node = None
            for v in V:

                if dist[u][v] < min_dist:
                    min_dist = dist[u][v]
                    min_node = v

            # Get the cluster of the closest node
            r = v2c[min_node]
            Q[r].append(u)

    # Add the new nodes to the original partition
    R = copy.copy(P)
    for cluster, nodes in Q.items():
        R[cluster].extend(nodes)

    return R


def obi_main(C, k, method):
    """Main OBI function that returns a k-partiton given some clustering method."""
    rg_sg = compute_rg_sg(C, k, methods=[method])
    P = _Partition({i: v for i, v in enumerate(rg_sg.vs["name"])})
    Q = extend_partition(C.G, P)
    return Partition(Q)
