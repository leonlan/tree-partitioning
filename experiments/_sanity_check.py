from tree_partitioning.classes import ReducedGraph
from tree_partitioning.utils import remove_lines


def _sanity_check(G, generators, partition, lines):
    """
    Check if all the following conditions are satisfied:
    - The partition has connected clusters;
    - Each generator group belongs to one cluster;
    - The post switching graph has correct number of edges/nodes;
    - The post switching reduced graph is a tree.
    """
    assert partition.is_connected_clusters(G)

    for gens in generators.values():

        contained = False
        for cluster in partition.clusters.values():
            if set(gens).intersection(set(cluster)) == set(gens):
                contained = True
                break

        if not contained:
            assert False

    G_post = remove_lines(G, lines)[0]  # First connected component
    assert len(G.edges) == len(G_post.edges) + len(lines)
    assert len(G.nodes) == len(G_post.nodes)

    rg = ReducedGraph(G_post, partition)
    assert rg.is_tree()
    assert len(rg.RG.edges) == len(generators) - 1
