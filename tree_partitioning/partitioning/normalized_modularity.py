from .legacy import obi_main


def normalized_modularity(k):
    return obi_main(k, method="ModularityN")
