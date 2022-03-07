#!/usr/bin/env ipython
import igraph as ig
import numpy as np
import scipy
import sklearn

from tree_partitioning.classes import Partition


def spectral_clustering(
    igg, n_clusters: int, matrix: str = "Laplacian", weight: str = "weight"
):
    """
    Spectral clustering functions for various matrix definitions.

    Args:
    - igg: igraph
    - n_clusters: Number of target clusters
    - matrix: Matrix variant used for clustering (default: Laplacian)
    - weight: Weight attribute (default: weight)

    Returns:
    - partition: Partition object

    """
    M = _compute_matrix(igg, matrix, weight)

    # Calculate the eigenspectrum  of the matrix sorted in non-increasing order
    w, v = scipy.linalg.eigh(M)

    # Keep only the n_clusters eigenvectors corresponding to the two smallest eigenvalues
    # for the Laplacian matrices and the two largest for the modularity matrices
    if matrix.startswith("Laplacian"):
        X = v[:, :n_clusters]
    else:
        X = v[:, -n_clusters:]

    # Construct matrix Y by renormalizing X
    norm_matrix = np.reshape(np.linalg.norm(X, axis=1), (X.shape[0], 1))
    Y = np.divide(X, norm_matrix, out=np.zeros_like(X), where=norm_matrix != 0)

    # Cluster rows of Y into k clusters using K-means
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(Y)

    # Assign original point i to the cluster of the row i of matrix Y
    clusters = np.array(kmeans.labels_)
    return Partition.from_clustering(ig.VertexClustering(igg, membership=clusters))


def _compute_matrix(igg, matrix: str, weight: str):
    """
    Util to return the input matrix
    """
    if matrix == "LaplacianN":
        return IGnormalized_laplacian(igg, weight=weight).todense()

    elif matrix == "ModularityN":
        return IGnormalized_modularity_matrix(igg, weight=weight).todense()

    elif matrix == "Laplacian":
        return igg.laplacian(weights=weight)

    elif matrix == "Modularity":
        return IGmodularity_matrix(igg, weight=weight).todense()

    else:
        raise NameError("Given matrix name does not exists.")


def IGnormalized_modularity_matrix(G, weight=None):
    """Computes the normalized modularity matrix."""
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
    """
    Computes the normalized Laplacian.

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
    """
    Computes the modularity matrix.

    Formula (entry-wise):
    G_{ij} = A_{ij} - (k_i*k_j)/(2m)
    """
    A = G.get_adjacency_sparse(attribute=weight)
    k = A.sum(axis=1)
    M = k.sum() * 0.5
    X = scipy.sparse.csr_matrix(k * k.transpose() / (2 * M))
    B = A - X
    return B
