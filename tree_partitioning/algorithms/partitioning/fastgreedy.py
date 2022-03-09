from tree_partitioning.classes import Partition


def fastgreedy(igg, n_clusters: int, weights: str = "weight"):
    """
    Cluster the graph using fastgreedy
    """
    # The induced subgraph corresponding to the biggest bridge-block gets split into n_clusters
    simple_igg = igg.simplify(combine_edges=dict(weight="sum"))
    partition = simple_igg.community_fastgreedy(weights=weights).as_clustering(
        n=n_clusters
    )

    return Partition.from_clustering(simple_igg, partition)
