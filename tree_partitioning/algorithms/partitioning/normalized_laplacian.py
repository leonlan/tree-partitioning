from .legacy import obi_main


def normalized_laplacian(k):
    return obi_main(k, method="LaplacianN")
