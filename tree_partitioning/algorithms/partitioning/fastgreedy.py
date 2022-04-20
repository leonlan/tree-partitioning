from .legacy import obi_main


def fastgreedy(n_clusters: int):
    return obi_main(n_clusters, method="FastGreedy")
