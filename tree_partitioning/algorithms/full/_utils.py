def _partitioning_alg_selection(algorithm: str):
        case "spectral_clustering"
    partitioning_alg = (
        spectral_clustering if partitioning == "spectral_clustering" else fastgreedy
    )


def _line_switching_alg_selection():

    line_switching_alg = (
        milp_line_switching if line_switching == "milp_line_switching" else brute_force
    )
