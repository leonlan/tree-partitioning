# import pulp as pl
import igraph as ig
import copy
from collections import defaultdict
from itertools import product


import igraph as ig
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.converter as pc
import pandapower.networks as pn
import sklearn
from sklearn.cluster import SpectralClustering
import scipy
import scipy.sparse

from utils import permute_matrix


def bridge_util_simple(igg, u, visited, parent, low, disc, time, bridgelist):

    # Mark the current node as visited and print it
    visited[u] = True

    # Initialize discovery time and low value
    disc[u] = time
    low[u] = time
    time += 1

    # Recur for all the vertices adjacent to this vertex
    for v in igg.neighbors(u):

        # If v is not visited yet, then make it a child of u in DFS tree and recur for it
        if visited[v] == False:
            parent[v] = u
            visited, parent, low, disc, time, bridgelist = bridge_util(
                igg, v, visited, parent, low, disc, time, bridgelist
            )

            # Check if the subtree rooted with v has a connection to one of the ancestors of u
            low[u] = min(low[u], low[v])

            # If the lowest vertex reachable from subtree under v is below u in DFS tree, then u-v is a bridge
            if low[v] > disc[u]:
                #                 print('Bridge found between vertices idx ',u," and ",v,' edge=(',igg.vs['name'][u],',',igg.vs['name'][v],')')
                bridgelist.append((u, v))

        elif v != parent[u]:  # Update low value of u for parent function calls.
            low[u] = min(low[u], disc[v])

    return visited, parent, low, disc, time, bridgelist


# DFS based function to find all bridges. It uses recursive function bridgeUtil()
def bridges_simple(igg):

    # Mark all the vertices as not visited and Initialize parent and visited arrays
    visited = [False] * (igg.vcount())  # keeps tract of visited vertices
    disc = [float("Inf")] * (igg.vcount())  # Stores discovery times of visited vertices
    low = [float("Inf")] * (
        igg.vcount()
    )  # stores lowest reachable vertex from subtree under a given now
    parent = [-1] * (igg.vcount())  # stores parent vertices in DFS tree
    time = 0
    bridgelist = []

    # Call the recursive helper function to find bridges in DFS tree rooted with vertex 'i'
    for v in igg.vs:
        if visited[v.index] == False:
            visited, parent, low, disc, time, bridgelist = bridge_util(
                igg, v.index, visited, parent, low, disc, time, bridgelist
            )

    return bridgelist


def bridge_util(igg, u, visited, parent, low, disc, time, bridgelist, incomingedge):
    """Recursive helper function for finding bridges."""

    # Mark the current node as visited and print it
    visited[u] = True

    # Initialize discovery time and low value
    disc[u] = time
    low[u] = time
    time += 1

    # Recur for all the vertices adjacent to this vertex
    for eidx in igg.incident(u, mode="all"):
        e = igg.es[eidx]
        if e.source == u:
            v = e.target
        else:
            v = e.source

        # If v is not visited yet, then make it a child of u in DFS tree and recur for it
        if visited[v] == False:
            parent[v] = u
            incomingedge[v] = eidx

            visited, parent, low, disc, time, bridgelist, incomingedge = bridge_util(
                igg, v, visited, parent, low, disc, time, bridgelist, incomingedge
            )

            # Check if the subtree rooted with v has a connection to one of the ancestors of u
            low[u] = min(low[u], low[v])

            # If the lowest vertex reachable from subtree under v is below u in DFS tree, then u-v is a bridge
            if low[v] > disc[u]:
                #                 print('Bridge found between vertices idx ',u," and ",v,' edge=(',igg.vs['name'][u],',',igg.vs['name'][v],')')
                bridgelist.append((u, v))

        elif (
            eidx != incomingedge[u]
        ):  # Update low value of u for parent function calls.
            low[u] = min(low[u], disc[v])

    return visited, parent, low, disc, time, bridgelist, incomingedge


# DFS based function to find all bridges. It uses recursive function bridgeUtil()
def bridges(igg):
    """Returns the list of bridges in igg."""

    # Mark all the vertices as not visited and initialize parent and visited arrays
    visited = [False] * (igg.vcount())  # Keeps track of visited vertices
    disc = [float("Inf")] * (igg.vcount())  # Stores discovery times of visited vertices
    low = [float("Inf")] * (
        igg.vcount()
    )  # Stores lowest reachable vertex from subtree under a given now
    parent = [-1] * (igg.vcount())  # Stores parent vertices in DFS tree
    time = 0
    bridgelist = []
    incomingedge = [-1] * (igg.vcount())  # Stores incoming edge for vertices

    # Call the recursive helper function to find bridges in DFS tree rooted with vertex 'i'
    for v in igg.vs:
        if visited[v.index] == False:
            visited, parent, low, disc, time, bridgelist, incomingedge = bridge_util(
                igg, v.index, visited, parent, low, disc, time, bridgelist, incomingedge
            )

    return bridgelist


def networkanalysis(igg, tableflag=False, verboseflag=False):
    """Analyzes the bridge block partition of the graph igg.

    - If verboseflag == True, print descriptive information about the bbd.
    - If tableflag == True, return the descriptive information.
    - Otherwise, return the bblocks partition and the bridge indices.
    """
    # Compute the block decomposition (i.e. block-cut-tree)
    blocks, ap = igg.biconnected_components(return_articulation_points=True)
    blocks_sizes = np.array([len(c) for c in blocks], dtype=np.int32)

    # Compute the initial bridge block decomposition
    bridgelist = bridges(igg)
    GG = igg.copy()
    GG.delete_edges(bridgelist)
    bblock_partition = GG.clusters()
    bblocks_sizes = np.array([len(c) for c in bblock_partition], dtype=np.int32)

    # Count number of trivial and non-trivial bblocks
    trivial_bblocks_count = (bblocks_sizes == 1).sum()
    nontrivial_bblocks_count = (bblocks_sizes != 1).sum()
    bblocks_count = len(bblocks_sizes)

    # Print descriptive information
    if verboseflag:
        print(
            "#ap =",
            len(ap),
            "#blocks =",
            len(blocks),
            "#bridges (guess)=",
            sum([len(b) == 2 for b in blocks]),
            "sizes=",
            [len(c) for c in blocks],
        )
        print(
            "#bridges =",
            len(bridgelist),
            "#bridge-blocks=",
            bblocks_count,
            "#nontrivial=",
            nontrivial_bblocks_count,
            "#trivial=",
            trivial_bblocks_count,
            "sizes=",
            list(np.sort(bblocks_sizes)),
        )

    # Return descriptive information
    if tableflag:
        return [
            0.0,
            igg.vcount(),
            igg.ecount(),
            igg.has_multiple(),
            0.0,
            0.0,
            0,
            len(ap),
            len(blocks),
            np.sort(blocks_sizes)[::-1],
            len(bridgelist),
            bblocks_count,
            nontrivial_bblocks_count,
            trivial_bblocks_count,
            np.sort(bblocks_sizes)[::-1],
        ]

    # Return the bblock partition and the edge indices of the bridges
    else:
        return bblock_partition, list(itertools.starmap(igg.get_eid, bridgelist))


def oneshotalgorithm(
    net,
    igg,
    n_clusters,
    goal="max",
    methods=[
        # "Laplacian",
        # "Modularity",
        # "LaplacianRW",
        # "LaplacianN",
        # "ModularityN",
        "FastGreedy"
    ],
):
    """Bridge block refinement procedure using the one shot algorithm.

    The one shot algorithm solves the OBI-k and OSP seperately, where the OSP
    problem is solved using total enumeration.

    Args:
    - net: pandapower network
    - igg: iGraph graph
    - n_clusters: Number of target bridge blocks
    - goal: Optimization goal (default: max)
    - methods: List of methods for spectral clustering (default: all)

    Returns:
    - ...
    """
    # Obtain the initial BBD
    block_partition, bridgeidx = networkanalysis(igg, verboseflag=False)

    C = []

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
        reducedgraph.contract_vertices(npartition.membership, combine_attrs="min")
        reducedgraph.simplify(multiple=False, loops=True)
        # ig.plot(reducedgraph, vertex_label=reducedgraph.vs["community"])

        # Focuses on the subgraph of the reduced graph where the pruning is needed
        rg_sg = reducedgraph.subgraph(sgcommunities)

        # TODO: Determine if this needs to be kept
        #         print('Method',i,'selected a bridge-block of size',lbbsize,'and splitted into',len(sgcommunities),'smaller bridge-blocks')
        #         if not(connectedpartition_flag):
        #             print('These are more than',n_clusters,' since the partition had disconnected clusters')
        nce, nst, bs, maxc, avgc, ncl, rlidx, rlt = smartoptimalswitching(
            rg_sg, net, goal, verboseflag=False
        )

        elapsedt = time.time() - t

        results.loc[method] = pd.Series(
            {
                "runtime": elapsedt,
                "modularity": Q,
                "cross_edges": nce,
                "spanning_trees": nst,
                "lbbsize": lbbsize,
                "#active_lines": activelines - len(rlidx),
                "bsizes_split": sorted(bsplit_pre),
                "bcount": len(bsize_post),
                "block_sizes_post": sorted(bsize_post, reverse=True),
                "max_cong": maxc,
                "avg_cong": avgc,
                "#congested_lines": ncl,
                "best_scenario": bs,
                "#removed_lines": len(rlidx),
                "percentage_removed_lines": 100 * len(rlidx) / igg.ecount(),
                "removed_lines_type": rlt,
            }
        )

    print(
        "\nOne-shot split into",
        n_clusters,
        f"clusters with criterium",
        goal,
        "using various clustering algorithms",
    )

    return results


def smartoptimalswitching(rg_sg, net, goal="max", verboseflag=False):
    """Solves the optimal swithing problem for the reduced graph rg_sg using brute-force.

    Args:
    - rg_sg: The reduced graph induced by the potential clusters
    - net: pandapower format network
    - goal: Optimization goal (defautl: max)

    Returns:
    - TODO
    """

    # Run an initial DC power flow
    pp.rundcpp(net)
    initialcongestion = net.res_line["loading_percent"]
    initialcongestion = initialcongestion.combine_first(
        net.res_trafo["loading_percent"]
    )
    initialcongestion.fillna(
        0, inplace=True
    )  # Fixing the possible NaN corresponding to the lines with zero flow

    # Brute-force through all possible spanning trees
    c = 0  # Number of spanning trees considered
    for removededges in [
        list(i)
        for i in itertools.combinations(
            [e.index for e in rg_sg.es], rg_sg.ecount() - rg_sg.vcount() + 1
        )
    ]:

        # Remove potential edges and check if the rg_sg is still connected
        l = rg_sg.copy()
        l.delete_edges(removededges)
        if l.is_connected():
            c += 1
            # Remove the edges from the original network
            net_temp = pp.copy.deepcopy(net)  # Temporary net
            for reidx in removededges:
                if rg_sg.es["type"][reidx] == "L":
                    net_temp.line.drop(rg_sg.es["index_by_type"][reidx], inplace=True)
                else:
                    net_temp.trafo.drop(rg_sg.es["index_by_type"][reidx], inplace=True)

            # Run a DC power flow
            pp.rundcpp(net_temp)
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])
            congestion.fillna(0, inplace=True)
            if verboseflag:
                print(
                    "Scenario ",
                    c,
                    "\nRemoved lines ",
                    [reducedgraph.es["index"][reidx] for reidx in removededges],
                )
                print(
                    "Max congestion= ",
                    max(congestion) / 100,
                    " Mean congestion =",
                    sum(congestion) / (100 * len(congestion)),
                    " # congested lines=",
                    (congestion >= 99.99).sum(),
                )
            # TODO: Define an update function for bestscenario, bestset, bestcongestion
            if c == 1:
                bestscenario = c
                bestset = removededges
                bestcongestion = congestion
            elif goal == "max" and max(congestion) <= max(bestcongestion):
                if (
                    max(congestion) == max(bestcongestion)
                    and (congestion >= 99.999).sum() == (bestcongestion >= 99.999).sum()
                ):
                    if (congestion >= 99.999).sum() < (bestcongestion >= 99.999).sum():
                        bestscenario = c
                        bestset = removededges
                        bestcongestion = congestion
                if (
                    max(congestion) == max(bestcongestion)
                    and (congestion >= 99.999).sum() < (bestcongestion >= 99.999).sum()
                ):
                    bestscenario = c
                    bestset = removededges
                    bestcongestion = congestion
                if max(congestion) < max(bestcongestion):
                    bestscenario = c
                    bestset = removededges
                    bestcongestion = congestion
            elif (
                goal == "ncl"
                and (congestion >= 99.999).sum() <= (bestcongestion >= 99.999).sum()
            ):
                if (congestion >= 99.999).sum() == (
                    bestcongestion >= 99.999
                ).sum() and max(congestion) == max(bestcongestion):
                    if (congestion >= 99.999).sum() < (bestcongestion >= 99.999).sum():
                        bestscenario = c
                        bestset = removededges
                        bestcongestion = congestion
                if (congestion >= 99.999).sum() == (
                    bestcongestion >= 99.999
                ).sum() and max(congestion) < max(bestcongestion):
                    bestscenario = c
                    bestset = removededges
                    bestcongestion = congestion
                if (congestion >= 99.999).sum() < (bestcongestion >= 99.999).sum():
                    bestscenario = c
                    bestset = removededges
                    bestcongestion = congestion
    if verboseflag:
        print(
            "\nInitial scenario: Max congestion= ",
            max(initialcongestion) / 100,
            " Avg congestion =",
            sum(initialcongestion) / (100 * len(initialcongestion)),
            " # congested lines=",
            (initialcongestion >= 99.99).sum(),
            "\n",
        )
    print(sorted([rg_sg.es["name"][reidx] for reidx in bestset]))
    return (
        rg_sg.ecount(),
        c,
        bestscenario,
        max(bestcongestion) / 100,
        sum(bestcongestion) / (100 * len(bestcongestion)),
        (bestcongestion >= 99.99).sum(),
        [rg_sg.es["index_by_type"][reidx] for reidx in bestset],
        [rg_sg.es["type"][reidx] for reidx in bestset],
    )


def bridgeblockrefinement(
    net,
    igg,
    n_clusters=2,
    goal="max",
    method="LaplacianN",
    plotflag=False,
    fixedlayout=None,
):
    """Compute the bridge block refinement of the network net/graph igg.

    Args:
    - net: The network that is considered [TODO: Which data type?]
    - igg: iGraph of the network
    - goal: Optimization goal (default: max)
    - method: Clustering method (default: LaplacianN, normalized version)
    - plotflag: ??
    - fixedlayout: ??

    Returns:
    - ??
    """

    G = igg.copy()

    # find pre-existing bridges and bridge-blocks of the graph and the biggest one in size
    block_partition, bridgeidx = networkanalysis(G)

    if plotflag:
        ig.plot(
            ig.vertexclustering(g, membership=block_partition.membership),
            vertex_label=G.vs["name"],
            edge_label=G.vs["name"],
            layout=fixedlayout,
        ).show()

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
        sgpartition = spectralclustering(sg, n_clusters, matrix=method, weight="weight")

    # cluster sizes and modularity score of the selected partition of sg
    bsplit_pre = list(map(len, list(sgpartition)))
    Q = sg.modularity(sgpartition, weights="weight")

    # The original graph partition gets updated accordingly
    newm = block_partition.membership  # New membership vector for the full graph
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
            ]  # Find indices of nodes that belong to community c
            newcommunityindex = (
                max(newm) + 1
            )  # Community c inside subgraph gets updated index, which is equal to the current max index +1
            sgcommunities.append(newcommunityindex)
            for idx in [sg.vs["name"][sgidx] for sgidx in idxcommunity]:
                newm[
                    G.vs.select(name_eq=idx)[0].index
                ] = newcommunityindex  # All nodes that belong to community c gets updated memberhip in the full graph partition
    G.vs["community"] = newm
    npartition = ig.VertexClustering(
        G, membership=newm
    )  # new partition for the full graph

    # the following for loop deals with the selected partition possibly having disconnected components
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

    # color attribute is given to the graph nodes depending on the cluster they belong to
    igg.vs["community"] = newm
    igg.vs["color"] = ig.drawing.colors.ClusterColoringPalette(
        len(npartition)
    ).get_many(npartition.membership)

    if plotflag:
        ig.plot(
            npartition,
            layout=fixedlayout,
            vertex_label=igg.vs["community"],
            edge_label=igg.es["name"],
        ).show()
    # reduced graph obtained by first collapsing vertex in the same cluster and then removing loops
    # multiple edges are purposefully left
    # each bridge-block has a 'name' attribute equal to the lowest vertex 'name' that it contains
    # each bridge-block has the color of the nodes in the corresponding cluster
    reducedgraph = igg.copy()
    reducedgraph.contract_vertices(npartition.membership, combine_attrs="min")
    reducedgraph.simplify(multiple=False, loops=True)

    if plotflag:
        ig.plot(reducedgraph, vertex_label=reducedgraph.vs["community"]).show()

    # focuses on the subgraph of the reduced graph where the pruning is needed
    rg_sg = reducedgraph.subgraph(sgcommunities)

    #     print('Method',method,'selected a bridge-block of size',lbbsize,'and splitted into',len(sgcommunities),'smaller bridge-blocks')
    #     if not(connectedpartition_flag):
    #             print('These are more than',n_clusters,' since the partition had disconnected clusters')

    nce, nst, bs, maxc, avgc, ncl, rlidx, rlt = smartoptimalswitching(
        rg_sg, net, goal, verboseflag=False
    )

    return Q, maxc, avgc, ncl, nce, nst, lbbsize, bsplit_pre, bs, rlidx, rlt


def recursivealgorithm(
    net,
    igg,
    n_clusters=2,
    updateflows=False,
    goal="max",
    method="LaplacianN",
    steps=[0, 1, 2],
    plotflag=False,
):

    t = time.time()
    results = pd.DataFrame(
        columns=[
            "runtime",
            "modularity",
            "max_cong",
            "avg_cong",
            "#congested_lines",
            "cross_edges",
            "spanning_trees",
            "#removed_lines",
            "percentage_removed_lines",
            "#active_lines",
            "lbbsize",
            "target",
            "bcount",
            "block_sizes",
            "best_scenario",
            "removed_lines_type",
        ],
        index=steps,
    )

    G = igg.copy()
    net_temp = pp.copy.deepcopy(net)
    initiallayout = G.layout()

    for s in results.index:

        partialt = time.time()

        if s == 0:
            # bridge structure and congestion before any swithicing action
            pp.runpp(
                net_temp, init="results", max_iteration=100,
            )
            activelines = sum(net_temp.line["in_service"] == True) + sum(
                net_temp.trafo["in_service"] == True
            )
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])
            block_partition, bridgeidx = networkanalysis(G)
            partialelapsedt = time.time() - partialt
            results.loc[s] = pd.Series(
                {
                    "runtime": partialelapsedt,
                    "modularity": 0,
                    "cross_edges": 0,
                    "spanning_trees": 0,
                    "lbbsize": 0,
                    "target": [],
                    "bcount": len(list(block_partition)),
                    "block_sizes": sorted(
                        list(map(len, list(block_partition))), reverse=True
                    ),
                    "max_cong": max(congestion) / 100,
                    "avg_cong": sum(congestion) / (100 * len(congestion)),
                    "#congested_lines": (congestion >= 99.99).sum(),
                    "best_scenario": 0,
                    "#active_lines": activelines,
                    "#remove_lines": 0,
                    "percentage_removed_lines": 0.0,
                    "removed_lines_type": [],
                }
            )

        else:

            (
                Q,
                maxc,
                avgc,
                ncl,
                nce,
                nst,
                lbbsize,
                bsplit_pre,
                bs,
                rlidx,
                rlt,
            ) = bridgeblockrefinement(
                net_temp,
                G,
                n_clusters,
                goal,
                plotflag=plotflag,
                method=method,
                fixedlayout=initiallayout,
            )

            # update graph
            # by removing the lines with index rl
            # remove them

            # remove lines in net_temp
            true_eidx = []

            for reidx, ret in enumerate(rlt):
                if ret == "L":
                    #                     print('removing line',net_temp.line.at[rlidx[reidx],'name'])
                    true_eidx.append(
                        G.es.find(name=net_temp.line.at[rlidx[reidx], "name"]).index
                    )
                    net_temp.line.drop(rlidx[reidx], inplace=True)
                elif ret == "T":
                    #                     print('removing trafo',net_temp.trafo.at[rlidx[reidx],'name'])
                    true_eidx.append(
                        G.es.find(name=net_temp.trafo.at[rlidx[reidx], "name"]).index
                    )
                    net_temp.trafo.drop(rlidx[reidx], inplace=True)

            # remove edges from G
            G.delete_edges(true_eidx)

            # update flows and recalculate congestion
            pp.runpp(net_temp, init="results", max_iteration=100)
            activelines = sum(net_temp.line["in_service"] == True) + sum(
                net_temp.trafo["in_service"] == True
            )
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])

            if updateflows:
                # update edge attributes in view of the new flows and congestions
                for idx, row in net_temp.line.iterrows():
                    G.es.find(name=row["name"])["weight"] = abs(
                        net_temp.res_line.at[idx, "p_from_mw"]
                    )
                    G.es.find(name=row["name"])["loading_percent"] = abs(
                        net_temp.res_line.at[idx, "loading_percent"]
                    )
                for idx, row in net_temp.trafo.iterrows():
                    G.es.find(name=row["name"])["weight"] = abs(
                        net_temp.res_trafo.at[idx, "p_hv_mw"]
                    )
                    G.es.find(name=row["name"])["loading_percent"] = abs(
                        net_temp.res_trafo.at[idx, "loading_percent"]
                    )

            partialelapsedt = time.time() - partialt
            block_partition, bridgeidx = networkanalysis(G)

            #             if plotflag:
            #                 ig.plot(ig.VertexClustering(G, membership = block_partition.membership), layout=initiallayout, edge_label=G.es['name']).show()

            results.loc[s] = pd.Series(
                {
                    "runtime": partialelapsedt,
                    "modularity": Q,
                    "cross_edges": nce,
                    "spanning_trees": nst,
                    "lbbsize": lbbsize,
                    "target": bsplit_pre,
                    "bcount": len(list(block_partition)),
                    "block_sizes": sorted(
                        list(map(len, list(block_partition))), reverse=True
                    ),
                    "max_cong": max(congestion) / 100,
                    "avg_cong": sum(congestion) / (100 * len(congestion)),
                    "#congested_lines": (congestion >= 99.99).sum(),
                    "best_scenario": bs,
                    "#active_lines": activelines,
                    "#removed_lines": len(true_eidx),
                    "percentage_removed_lines": 100 * len(true_eidx) / igg.ecount(),
                    "removed_lines_type": rlt,
                }
            )

    elapsedt = time.time() - t

    print(
        "\nRecursive split using method",
        method,
        "for",
        max(steps),
        "iterations with critierium",
        goal,
        "and updateflows=",
        updateflows,
        ". Total line removed=",
        sum(results["#removed_lines"]),
    )

    return results


def greedyrecursivealgorithm(
    net,
    igg,
    n_clusters=2,
    updateflows=False,
    goal="max",
    methodlist=["LaplacianN", "ModularityN", "FastGreedy"],
    steps=[0, 1, 2],
    plotflag=False,
):
    """Compute the bridge block refinement using the recursive approach.

    The recursive algorithm applies bisective recursion to the largest bridge block until
    the desired number of bridge blocks are obtained.

    Args:
    - net: pandapower network
    - igg: igraph of the network
    - n_clusters: Number of target clusters
    - updateflows: ???
    - goal: Optimization goal (default: max)
    - methodlist: ???
    - steps: ???
    - plotflag: ???

    Returns
    -
    """

    # Define tracking variables
    t = time.time()
    results = pd.DataFrame(
        columns=[
            "runtime",
            "modularity",
            "max_cong",
            "avg_cong",
            "#congested_lines",
            "cross_edges",
            "spanning_trees",
            "#removed_lines",
            "#active_lines",
            "lbbsize",
            "target",
            "bcount",
            "block_sizes",
            "best_scenario",
            "removed_lines_type",
            "method",
        ],
        index=steps,
    )

    G = igg.copy()
    net_temp = pp.copy.deepcopy(net)
    initiallayout = G.layout()

    # Run len(steps) iterations of the recursive algorithm
    for s in results.index:

        partialt = time.time()

        if s == 0:
            # Compute bridge structure and congestion before any switching action
            pp.rundcpp(net_temp)
            activelines = sum(net_temp.line["in_service"] == True) + sum(
                net_temp.trafo["in_service"] == True
            )
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])
            block_partition, bridgeidx = networkanalysis(G)
            partialelapsedt = time.time() - partialt
            results.loc[s] = pd.Series(
                {
                    "runtime": partialelapsedt,
                    "modularity": 0,
                    "cross_edges": 0,
                    "spanning_trees": 0,
                    "lbbsize": 0,
                    "target": [],
                    "bcount": len(list(block_partition)),
                    "block_sizes": sorted(
                        list(map(len, list(block_partition))), reverse=True
                    ),
                    "max_cong": max(congestion) / 100,
                    "avg_cong": sum(congestion) / (100 * len(congestion)),
                    "#congested_lines": (congestion >= 99.99).sum(),
                    "best_scenario": 0,
                    "#active_lines": activelines,
                    "#removed lines": 0,
                    "removed_lines_type": [],
                    "method": "",
                }
            )

        else:
            # Initial values
            bestmaxc = 10
            bestncl = 1000
            bestmethod = "LaplacianN"

            for currentmethod in methodlist:
                (
                    Q,
                    maxc,
                    avgc,
                    ncl,
                    nce,
                    nst,
                    lbbsize,
                    bsplit_pre,
                    bs,
                    rlidx,
                    rlt,
                ) = bridgeblockrefinement(
                    net_temp, G, n_clusters, goal, method=currentmethod, plotflag=False
                )

                if goal == "max" and maxc <= bestmaxc:
                    if maxc == bestmaxc and ncl == bestncl:
                        if ncl < bestncl:
                            bestmethod = currentmethod
                            bestmaxc = maxc
                            bestncl = ncl
                    if maxc == bestmaxc and ncl < bestncl:
                        bestmethod = currentmethod
                        bestmaxc = maxc
                        bestncl = ncl
                    if maxc < bestmaxc:
                        bestmethod = currentmethod
                        bestmaxc = maxc
                        bestncl = ncl
                elif goal == "ncl" and ncl <= bestncl:
                    if ncl == bestncl and maxc == bestmaxc:
                        if ncl < bestncl:
                            bestmethod = currentmethod
                            bestmaxc = maxc
                            bestncl = ncl
                    if ncl == bestncl and maxc < bestmaxc:
                        bestmethod = currentmethod
                        bestmaxc = maxc
                        bestncl = ncl
                    if ncl < bestncl:
                        bestmethod = currentmethod
                        bestmaxc = maxc
                        bestncl = ncl

            (
                Q,
                maxc,
                avgc,
                ncl,
                nce,
                nst,
                lbbsize,
                bsplit_pre,
                bs,
                rlidx,
                rlt,
            ) = bridgeblockrefinement(
                net_temp,
                G,
                n_clusters,
                goal,
                plotflag=plotflag,
                method=bestmethod,
                fixedlayout=initiallayout,
            )

            # update graph
            # by removing the lines with index rl
            # remove them

            # Remove lines in net_temp
            true_eidx = []

            for reidx, ret in enumerate(rlt):
                if ret == "L":
                    #                     print('removing line',net_temp.line.at[rlidx[reidx],'name'])
                    true_eidx.append(
                        G.es.find(name=net_temp.line.at[rlidx[reidx], "name"]).index
                    )
                    net_temp.line.drop(rlidx[reidx], inplace=True)
                elif ret == "T":
                    #                     print('removing trafo',net_temp.trafo.at[rlidx[reidx],'name'])
                    true_eidx.append(
                        G.es.find(name=net_temp.trafo.at[rlidx[reidx], "name"]).index
                    )
                    net_temp.trafo.drop(rlidx[reidx], inplace=True)

            # Remove edges from G
            G.delete_edges(true_eidx)

            # Update flows and recalculate congestion
            pp.rundcpp(net_temp)
            activelines = sum(net_temp.line["in_service"] == True) + sum(
                net_temp.trafo["in_service"] == True
            )
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])

            if updateflows:
                # Update edge attributes in view of the new flows and congestions
                for idx, row in net_temp.line.iterrows():
                    G.es.find(name=row["name"])["weight"] = abs(
                        net_temp.res_line.at[idx, "p_from_mw"]
                    )
                    G.es.find(name=row["name"])["loading_percent"] = abs(
                        net_temp.res_line.at[idx, "loading_percent"]
                    )
                for idx, row in net_temp.trafo.iterrows():
                    G.es.find(name=row["name"])["weight"] = abs(
                        net_temp.res_trafo.at[idx, "p_hv_mw"]
                    )
                    G.es.find(name=row["name"])["loading_percent"] = abs(
                        net_temp.res_trafo.at[idx, "loading_percent"]
                    )

            partialelapsedt = time.time() - partialt
            block_partition, bridgeidx = networkanalysis(G)

            #             if plotflag:
            #                 ig.plot(ig.VertexClustering(G, membership = block_partition.membership), layout=initiallayout, edge_label=G.es['name']).show()

            results.loc[s] = pd.Series(
                {
                    "runtime": partialelapsedt,
                    "modularity": Q,
                    "cross_edges": nce,
                    "spanning_trees": nst,
                    "lbbsize": lbbsize,
                    "target": bsplit_pre,
                    "bcount": len(list(block_partition)),
                    "block_sizes": sorted(
                        list(map(len, list(block_partition))), reverse=True
                    ),
                    "max_cong": max(congestion) / 100,
                    "avg_cong": sum(congestion) / (100 * len(congestion)),
                    "#congested_lines": (congestion >= 99.99).sum(),
                    "best_scenario": bs,
                    "#active_lines": activelines,
                    "#removed_lines": len(true_eidx),
                    "removed_lines_type": rlt,
                    "method": bestmethod,
                }
            )

    elapsedt = time.time() - t

    print(
        "\n Greedy recursive split using at each of the",
        max(steps),
        "iterations the best method with critierium",
        goal,
        "and updateflows=",
        updateflows,
        ". Total line removed=",
        sum(results["#removed lines"]),
    )

    return results


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


class Partition(dict):
    """Vertex partition class.

    Standard representation is a dictionary with clusters as keys and
    a list of vertex ids as values.
    """

    def __init__(self, *arg, **kw):
        super(Partition, self).__init__(*arg, **kw)
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
    dist = dict(nx.all_pairs_shortest_path_length(G))
    # breakpoint()
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


def spectral_clustering(L, k, normalized=True):
    """Cluster the Laplacian matrix using spectral clustering.

    Normalized spectral clustering by Ng, Jordan and Weiss (2002).
    The algorithm can be found in von Luxenburg (2007).

    :param L: Laplacian matrix
    :param k: Number of clusters
    :param normalized: Bool to indicate normalization of Laplacian
    """
    if normalized:
        Dsqrt = np.sqrt(np.diag(1 / np.diag(L)))
        L = Dsqrt @ L @ Dsqrt

    # Compute eigenproblem and sort in increasing order
    l, v = np.linalg.eig(L)
    idx = l.argsort()
    l = l[idx]
    v = v[:, idx]

    # Obtain the first k eigenvectors and normalize row norm 1
    U = v[:, :k]
    rows = np.array([np.sqrt(np.sum(U ** 2, axis=1))]).T
    T = U / rows
    # breakpoint()
    # Cluster the rows
    labels = KMeans(k).fit(T).labels_

    return labels


def obi_main(C, k, method):
    """Main OBI function that returns a k-partiton given some clustering method."""
    rg_sg = compute_rg_sg(C, k, methods=[method])
    P = Partition({i: v for i, v in enumerate(rg_sg.vs["name"])})
    Q = extend_partition(C.G, P)
    return Q


def constrained_spectral(L, P, normalized=False):
    """Constrained spectral clustering based on Quiros-Tortos (2015).

    Args:
    - L: Graph Laplacian
    - P: Partition class with coherent generators
    """
    # Initialize
    PP = P.vertex2cluster()
    k = len(P.keys())

    if normalized:
        Dsqrt = np.sqrt(np.diag(1 / np.diag(L)))
        L = Dsqrt @ L @ Dsqrt

    # Compute the eigenvectors
    l, v = np.linalg.eig(L)
    idx = l.argsort()
    l = l[idx]
    v = v[:, idx]
    # Obtain the first k eigenvectors and normalize row norm 1
    X = v[:, :k]
    rows = np.array([np.sqrt(np.sum(X ** 2, axis=1))]).T
    Y = X / rows

    # Select the centroids (generators) and the non-centroids (loads)
    centroids = sorted(list(PP.keys()))
    non_centroids = [i for i in range(len(L)) if i not in centroids]

    # Compute for each load the closest centroid
    centroids_groups = defaultdict(list)
    D = cdist(Y[non_centroids, :], Y[centroids, :])
    for i, bus in enumerate(non_centroids):
        closest_centroid = centroids[np.argmin(D[i])]
        centroids_groups[closest_centroid].append(bus)

    # Group each group of centroids
    Q = {k: v for k, v in P.items()}
    for centroid, group in centroids_groups.items():
        r = PP[centroid]  # Original cluster the centroid belongs to
        Q[r].extend(group)  # Extend the cluster groups

    return Partition(Q)
