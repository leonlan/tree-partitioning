import itertools
import time

import igraph as ig
import networkx as nx
import numpy as np
import pandapower as pp
import pandas as pd
import scipy
import scipy.sparse
import sklearn


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
