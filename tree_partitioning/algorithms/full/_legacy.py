from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
import itertools
import warnings
import pandapower as pp
import pandapower.converter as pc
import pandapower.networks as pn
import logging
import os
from sklearn.cluster import SpectralClustering
import scipy
import scipy.sparse
import networkx.algorithms.community as nx_comm
import sklearn
import math
import time


logging.basicConfig(level=logging.ERROR)


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


def normalized_modularity_matrix(G, nodelist=None, weight=None):
    """Computes the normalized modularity matrix of G.
    B = A - X
    TODO: What is computed here?
    """
    if nodelist is None:
        nodelist = G.nodes()
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, format="csr")
    k = A.sum(axis=1)
    M = k.sum() * 0.5
    X = scipy.sparse.csr_matrix(k * k.transpose() / (2 * M))
    B = A - X
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], m, n, format="csr")
    with scipy.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.lib.scimath.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
    return DH.dot(B.dot(DH))


def IGnormalized_modularity_matrix(G, weight=None):
    """Computes the normalized modularity matrix using igg."""
    A = G.get_adjacency_sparse(attribute=weight)
    k = A.sum(axis=1)
    M = k.sum() * 0.5
    X = scipy.sparse.csr_matrix(k * k.transpose() / (2 * M))
    B = A - X
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], m, n, format="csr")
    with scipy.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.lib.scimath.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
    return DH.dot(B.dot(DH))


def IGnormalized_laplacian(G, weight=None):
    """Computes the normalized Laplacian.

    Formula:
    LN = D^{-1/2} * (D - A) * D^{-1/2}
    """
    A = G.get_adjacency_sparse(attribute=weight)
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], m, n, format="csr")
    L = D - A
    with scipy.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.lib.scimath.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
    return DH.dot(L.dot(DH))


def IGmodularity_matrix(G, weight=None):
    """Computes the modularity matrix.

    Formula (entry-wise):
    G_{ij} = A_{ij} - (k_i*k_j)/(2m)
    """
    A = G.get_adjacency_sparse(attribute=weight)
    k = A.sum(axis=1)
    M = k.sum() * 0.5
    X = scipy.sparse.csr_matrix(k * k.transpose() / (2 * M))
    B = A - X
    return B


def IGlaplacian_RW(G, weight=None):
    """Computes ...
    TODO: Figure out what is computed here.
    """
    A = G.get_adjacency_sparse(attribute=weight)
    n, m = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], m, n, format="csr")
    with scipy.errstate(divide="ignore"):
        diags_inv = 1.0 / diags
    diags_inv[np.isinf(diags_inv)] = 0
    DH = scipy.sparse.spdiags(diags_inv, [0], m, n, format="csr")
    return scipy.sparse.eye(m, n) - DH.dot(A)


def spectralclustering(igg, n_clusters, matrix="Laplacian", weight="weight"):
    """Spectral clustering functions for various matrix definitions.

    Args:
    - igg: iGraph
    - n_clusters: Number of target clusters
    - matrix: Matrix variant used for clustering (default: Laplacian)
    - weight: Weight attribute (default: weight)

    Returns:
    - ??

    """
    if matrix == "LaplacianN":
        M = IGnormalized_laplacian(igg, weight=weight).todense()
    if matrix == "LaplacianRW":
        M = IGlaplacian_RW(igg, weight=weight).todense()
    if matrix == "ModularityN":
        M = IGnormalized_modularity_matrix(igg, weight=weight).todense()
    if matrix == "Laplacian":
        M = igg.laplacian(weights=weight)
    if matrix == "Modularity":
        M = IGmodularity_matrix(igg, weight=weight).todense()

    # Calculate the eigenspectrum  of the Laplacian matrix sorted in non-increasing order
    w, v = scipy.linalg.eigh(M)

    # Keeps only the n_clusters eigenvectors corresponding to the two smallest eingenvalues
    # for the Laplacian matrices and the two largest for the modularity matrices
    if matrix == "LaplacianN" or matrix == "Laplacian" or matrix == "LaplacianRW":
        X = v[:, :n_clusters]
    else:
        X = v[:, -n_clusters:]

    # Construct matrix Y by renormalizing X
    norm_matrix = np.reshape(np.linalg.norm(X, axis=1), (X.shape[0], 1))
    Y = np.divide(
        X, norm_matrix, out=np.zeros_like(X), where=norm_matrix != 0
    )  # the alternative was giving NaN errors, namely Y = np.divide(X, np.reshape(np.linalg.norm(X, axis=1), (X.shape[0], 1)))

    # Cluster rows of Y into k clusters using K-means
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(Y)

    # Assign original point i to the cluster of the row i of matrix Y
    clusters = np.array(kmeans.labels_)
    partition = ig.VertexClustering(igg, membership=clusters)
    return partition


def _change_ref_bus(net, ref_bus_idx, ext_grid_p=0):  # Copied from pandapower
    """
    This function changes the current reference bus / buses, declared by net.ext_grid.bus towards the given 'ref_bus_idx'. If ext_grid_p is a list, it must be in the same order as net.ext_grid.index.
    """
    # cast ref_bus_idx and ext_grid_p as list
    if not isinstance(ref_bus_idx, list):
        ref_bus_idx = [ref_bus_idx]
    if not isinstance(ext_grid_p, list):
        ext_grid_p = [ext_grid_p]
    for i in ref_bus_idx:
        if i not in net.gen.bus.values and i not in net.ext_grid.bus.values:
            raise ValueError("Index %i is not in net.gen.bus or net.ext_grid.bus." % i)
    # determine indices of ext_grid and gen connected to ref_bus_idx
    gen_idx = net.gen.index[net.gen.bus.isin(ref_bus_idx)]
    ext_grid_idx = net.ext_grid.index[~net.ext_grid.bus.isin(ref_bus_idx)]
    # old ext_grid -> gen
    j = 0
    for i in ext_grid_idx:
        ext_grid_data = net.ext_grid.loc[i]
        net.ext_grid.drop(i, inplace=True)
        pp.create_gen(
            net,
            ext_grid_data.bus,
            ext_grid_p[j],
            vm_pu=ext_grid_data.vm_pu,
            controllable=True,
            min_q_mvar=ext_grid_data.min_q_mvar,
            max_q_mvar=ext_grid_data.max_q_mvar,
            min_p_mw=ext_grid_data.min_p_mw,
            max_p_mw=ext_grid_data.max_p_mw,
        )
        j += 1
    # old gen at ref_bus -> ext_grid (and sgen)
    for i in gen_idx:
        gen_data = net.gen.loc[i]
        net.gen.drop(i, inplace=True)
        if gen_data.bus not in net.ext_grid.bus.values:
            pp.create_ext_grid(
                net,
                gen_data.bus,
                vm_pu=gen_data.vm_pu,
                va_degree=0.0,
                min_q_mvar=gen_data.min_q_mvar,
                max_q_mvar=gen_data.max_q_mvar,
                min_p_mw=gen_data.min_p_mw,
                max_p_mw=gen_data.max_p_mw,
            )
        else:
            pp.create_sgen(
                net,
                gen_data.bus,
                p_mw=gen_data.p_mw,
                min_q_mvar=gen_data.min_q_mvar,
                max_q_mvar=gen_data.max_q_mvar,
                min_p_mw=gen_data.min_p_mw,
                max_p_mw=gen_data.max_p_mw,
            )


def compute_susceptances(net, df, susceptance_method="pandapower"):
    # Obtain the susceptances using either pandapower of pypower calculations
    if susceptance_method == "pandapower":
        # Line values
        b_lines = np.array(
            1
            / (
                net.line["x_ohm_per_km"]
                * net.line["length_km"]
                * net.sn_mva
                / net.line["parallel"]
            )
        )

        # Transformer susceptances
        zk = net.trafo["vk_percent"] / 100 * net.sn_mva / net.trafo["sn_mva"]
        rk = net.trafo["vkr_percent"] / 100 * net.sn_mva / net.trafo["sn_mva"]
        xk = np.array(zk * zk - rk * rk) ** (1 / 2)
        # Fill nans in tap_step_percent with 0
        net_trafo_tap_step_percent = net.trafo["tap_step_percent"].fillna(0)
        tapratiok = np.array(1 - net_trafo_tap_step_percent / 100)
        b_trafo = 1 / (xk * tapratiok)

        b = np.append(b_lines, b_trafo)

    elif susceptance_method == "pypower":
        ### Convert to pypower format
        from pandapower.pd2ppc import (
            _pd2ppc,
            _calc_pq_elements_and_add_on_ppc,
            _ppc2ppci,
        )

        ppc, ppci = _pd2ppc(net)

        #### Function: _rund_dc_pf(ppci)
        from pandapower.pypower.idx_bus import VA, GS
        from pandapower.pf.ppci_variables import (
            _get_pf_variables_from_ppci,
            _store_results_from_pf_in_ppci,
        )
        from pandapower.pypower.dcpf import dcpf
        from pandapower.pypower.makeBdc import makeBdc
        from numpy import pi, zeros, real, bincount

        (
            baseMVA,
            bus,
            gen,
            branch,
            ref,
            pv,
            pq,
            on,
            gbus,
            _,
            refgen,
        ) = _get_pf_variables_from_ppci(ppci)

        #### Function: makeBdc
        from pandapower.pypower.idx_brch import (
            F_BUS,
            T_BUS,
            BR_X,
            TAP,
            SHIFT,
            BR_STATUS,
        )

        stat = branch[:, BR_STATUS]  ## ones at in-service branches
        b = np.real(stat / branch[:, BR_X])  ## series susceptance

    return b


def smartloadcase(filename, verboseflag=False, opfflag=True, congestionprintflag=False):
    """Smart load a test case by returning the net, dg and igg.

    Changes refbus if needed.
    """
    net = pc.from_mpc(filename)

    # Manual change of ref bus for specific files
    if filename == "pglib_opf_case6515_rte":
        _change_ref_bus(net, 6171, ext_grid_p=2850.78)
    if filename == "pglib_opf_case6470_rte":
        net.ext_grid.loc[0, ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]] *= 2
        _change_ref_bus(net, 5988, ext_grid_p=[-169.41])
    if filename == "pglib_opf_case2848_rte":
        _change_ref_bus(net, 271, ext_grid_p=[44.01])
    if filename == "pglib_opf_case1888_rte":
        _change_ref_bus(net, 1246, ext_grid_p=[-89.5])

    dictline = {i: i for i in range(len(net.line))}
    dicttrafo = {i: len(net.line) + i for i in range(len(net.trafo))}

    # Run a DC (optimal) power flow and reindex the results
    pp.rundcpp(net)
    precongestion = net.res_line["loading_percent"].rename(index=dictline)
    precongestion = precongestion.combine_first(
        net.res_trafo["loading_percent"].rename(index=dicttrafo)
    )
    if opfflag:
        pp.rundcopp(net)
    opfcongestion = net.res_line["loading_percent"].rename(index=dictline)
    opfcongestion = opfcongestion.combine_first(
        net.res_trafo["loading_percent"].rename(index=dicttrafo)
    )

    # Intermediate print update
    if verboseflag:
        if opfflag:
            print(
                "pre-OPF congestion= ",
                max(precongestion) / 100,
                " Avg congestion =",
                sum(precongestion) / (100 * len(precongestion)),
                " # congested lines=",
                (precongestion >= 99.99).sum(),
            )
        print(
            "postOPF congestion= ",
            max(opfcongestion) / 100,
            " Avg congestion =",
            sum(opfcongestion) / (100 * len(opfcongestion)),
            " # congested lines=",
            (opfcongestion >= 99.99).sum(),
        )

    # Change load/generator power injections setpoints according to OPF
    net.load["p_mw"] = net.res_load["p_mw"]
    net.gen["p_mw"] = net.res_gen["p_mw"]
    for i in net.line.index:
        net.line.at[i, "name"] = "L" + str(i)
    for i in net.trafo.index:
        net.trafo.at[i, "name"] = "T" + str(i)

    # Create df of the network
    dfnetwork = net.line[["from_bus", "to_bus", "in_service", "name"]].rename(
        index=dictline
    )
    dfnetwork = dfnetwork.combine_first(
        net.trafo[["hv_bus", "lv_bus", "in_service", "name"]].rename(
            columns={"hv_bus": "from_bus", "lv_bus": "to_bus"}, index=dicttrafo
        )
    )
    dfnetwork["from_bus"] = dfnetwork["from_bus"].astype(int)
    dfnetwork["to_bus"] = dfnetwork["to_bus"].astype(int)
    d = net.res_line["p_from_mw"].rename(index=dictline)
    dfnetwork["weight"] = abs(
        d.combine_first(net.res_trafo["p_hv_mw"].rename(index=dicttrafo))
    )
    d = net.res_line["loading_percent"].rename(index=dictline)
    dfnetwork["loading_percent"] = d.combine_first(
        net.res_trafo["loading_percent"].rename(index=dicttrafo)
    )
    dfnetwork["type"] = "L"
    dfnetwork["index_by_type"] = 0
    dfnetwork["edge_index"] = 0
    dfnetwork["b"] = compute_susceptances(net, dfnetwork)
    dfnetwork["c"] = dfnetwork["weight"] / dfnetwork["loading_percent"] * 100
    dfnetwork["edge_id"] = tuple(
        zip(dfnetwork["from_bus"], dfnetwork["to_bus"])
    )  # directed edge id
    for i in range(len(net.line) + len(net.trafo)):
        if i < len(net.line):
            dfnetwork.at[i, "type"] = "L"
            dfnetwork.at[i, "index_by_type"] = i
            dfnetwork.at[i, "edge_index"] = i
        else:
            dfnetwork.at[i, "type"] = "T"
            dfnetwork.at[i, "index_by_type"] = i - len(net.line)
            dfnetwork.at[i, "edge_index"] = i
    #     dfnetwork.sort_values(by=['from_bus','to_bus'], inplace=True)

    # Drop lines not in service
    dfnetwork.drop(dfnetwork[dfnetwork["in_service"] == False].index, inplace=True)

    # Create igraph of the network
    igg = ig.Graph.TupleList(
        dfnetwork.itertuples(index=False),
        directed=False,
        vertex_name_attr="name",
        weights=False,
        edge_attrs=[
            "in_service",
            "name",
            "weight",
            "loading_percent",
            "type",
            "index_by_type",
            "edge_index",
            "b",
            "c",
            "edge_id",
        ],
    )
    igg.vs["community"] = [0] * (igg.vcount())

    if congestionprintflag:
        return (
            net,
            dfnetwork,
            igg,
            max(opfcongestion) / 100,
            sum(opfcongestion) / (100 * len(opfcongestion)),
            (opfcongestion >= 99.99).sum(),
        )
    else:
        return net, dfnetwork, igg


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
        t = time.perf_counter()
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

        elapsedt = time.perf_counter() - t

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

    time_line_switching = time.perf_counter()
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
        time.perf_counter() - time_line_switching,
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

    time_partitioning = time.perf_counter()
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

    time_partitioning = time.perf_counter() - time_partitioning

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

    (
        nce,
        nst,
        bs,
        maxc,
        avgc,
        ncl,
        rlidx,
        rlt,
        time_line_switching,
    ) = smartoptimalswitching(rg_sg, net, goal, verboseflag=False)

    return (
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
        time_partitioning,
        time_line_switching,
    )


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

    t = time.perf_counter()
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
            "time_partitioning",
            "time_line_switching",
        ],
        index=steps,
    )

    G = igg.copy()
    net_temp = pp.copy.deepcopy(net)
    initiallayout = G.layout()

    for s in results.index:

        partialt = time.perf_counter()

        if s == 0:
            # bridge structure and congestion before any swithicing action
            pp.rundcpp(
                net_temp, init="results", max_iteration=100,
            )
            activelines = sum(net_temp.line["in_service"] == True) + sum(
                net_temp.trafo["in_service"] == True
            )
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])
            block_partition, bridgeidx = networkanalysis(G)
            partialelapsedt = time.perf_counter() - partialt
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
                    "time_partitioning": 0,
                    "time_line_switching": 0,
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
                time_partitioning,
                time_line_switching,
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
            pp.rundcpp(net_temp, init="results", max_iteration=100)
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

            partialelapsedt = time.perf_counter() - partialt
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
                    "time_partitioning": time_partitioning,
                    "time_line_switching": time_line_switching,
                }
            )

    elapsedt = time.perf_counter() - t

    results["runtime"] = results["time_partitioning"] + results["time_line_switching"]

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
    results["n_clusters"] = results.index + 1

    return (
        results.groupby(lambda x: 1)
        .agg(
            {
                "runtime": sum,
                "modularity": max,
                "max_cong": list,
                "#congested_lines": sum,
                "cross_edges": sum,
                "spanning_trees": sum,
                "#removed_lines": sum,
                "percentage_removed_lines": sum,
                "#active_lines": max,
                "lbbsize": max,
                "block_sizes": list,
                "time_partitioning": sum,
                "time_line_switching": sum,
                "n_clusters": max,
            }
        )
        .T.to_dict()[1]
    )


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
    t = time.perf_counter()
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

        partialt = time.perf_counter()

        if s == 0:
            # Compute bridge structure and congestion before any switching action
            pp.rundcpp(net_temp)
            activelines = sum(net_temp.line["in_service"] == True) + sum(
                net_temp.trafo["in_service"] == True
            )
            congestion = net_temp.res_line["loading_percent"]
            congestion = congestion.combine_first(net_temp.res_trafo["loading_percent"])
            block_partition, bridgeidx = networkanalysis(G)
            partialelapsedt = time.perf_counter() - partialt
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
                    "time_partitioning": 0,
                    "time_line_switching": 0,
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
                    time_partitioning,
                    time_line_switching,
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

            partialelapsedt = time.perf_counter() - partialt
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
                    "time_partitioning": time_partitioning,
                    "time_line_switching": time_line_switching,
                }
            )

    elapsedt = time.perf_counter() - t

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


def compute_rg_sg(igg, n_clusters=4, goal="max", methods=["FastGreedy"]):
    """Bridge block refinement procedure using the one shot algorithm.

    The one shot algorithm solves the OBI-k and OSP seperately, where the OSP
    problem is solved using total enumeration.

    Args:
    - igg: iGraph graph
    - n_clusters: Number of target bridge blocks
    - goal: Optimization goal (default: max)
    - methods: List of methods for spectral clustering (default: all)

    Returns:
    - ...
    """
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
        t = time.perf_counter()
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


if __name__ == "__main__":
    net, df, igg = smartloadcase(goc300.path)
    rg_sg = compute_rg_sg(igg, 4)
    print(sorted(rg_sg.es["name"]))
    start = time.perf_counter()
    results = recursivealgorithm(
        net,
        igg,
        n_clusters=2,
        updateflows=False,
        goal="max",
        method="ModularityN",
        steps=[0, 1, 2, 3],
        plotflag=False,
    )

    print(time.perf_counter() - start)
