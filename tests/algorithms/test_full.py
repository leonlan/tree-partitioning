import pytest

from tree_partitioning.algorithms.line_switching import milp_line_switching, brute_force
from tree_partitioning.algorithms.partitioning import (
    normalized_modularity,
    normalized_laplacian,
    fastgreedy,
)
from tree_partitioning.algorithms.full import two_stage, recursive
from tree_partitioning.classes import Case


def cases():
    def make_callable_case(path, **kwargs):
        def f():
            return Case.from_file(path, **kwargs)

        return f

    paths = [
        # "data/pglib_opf_case14_ieee.mat",
        "data/pglib_opf_case24_ieee_rts.mat",
        "data/pglib_opf_case30_ieee.mat",
        "data/pglib_opf_case39_epri.mat",
        "data/pglib_opf_case57_ieee.mat",
        "data/pglib_opf_case73_ieee_rts.mat",
        "data/pglib_opf_case118_ieee.mat",
        "data/pglib_opf_case162_ieee_dtc.mat",
        "data/pglib_opf_case179_goc.mat",
        "data/pglib_opf_case200_activ.mat",
        "data/pglib_opf_case240_pserc.mat",
        "data/pglib_opf_case300_ieee.mat",
        "data/pglib_opf_case500_goc.mat",
        "data/pglib_opf_case500_goc_postopf.mat",
        "data/pglib_opf_case588_sdet.mat",
        # "data/pglib_opf_case793_goc.mat",
        # "data/pglib_opf_case1354_pegase.mat",
        # "data/pglib_opf_case1888_rte.mat",
        # "data/pglib_opf_case1888_rte_postopf.mat",
        # "data/pglib_opf_case1951_rte.mat",
        # "data/pglib_opf_case2000_goc.mat",
        # "data/pglib_opf_case2383wp_k.mat",
        # "data/pglib_opf_case2383wp_k_postopf.mat",
        # "data/pglib_opf_case2736sp_k.mat",
        # "data/pglib_opf_case2737sop_k.mat",
        # "data/pglib_opf_case2746wop_k.mat",
        # "data/pglib_opf_case2746wp_k.mat",
        # "data/pglib_opf_case2848_rte.mat",
        # "data/pglib_opf_case2869_pegase.mat",
    ]
    return [
        (
            make_callable_case(path),
            path.lstrip("data/pglib_opf_case").rstrip(".mat"),
            k,
            part_alg,
        )
        for path in paths
        for k in range(5, 6)
        for part_alg in [normalized_modularity, normalized_laplacian, fastgreedy]
    ]


# @pytest.mark.parametrize("case, name, k, part_alg", cases())
# def test_two_stage_milp(case, name, k, part_alg):
#     c = case()
#     solution = two_stage(k, "congestion", part_alg, milp_line_switching, results=True)
#     assert solution.is_tree_partition()


# @pytest.mark.parametrize("case, name, k, part_alg", cases())
# def test_two_stage_bf(case, name, k, part_alg):
#     c = case()
#     solution = two_stage(k, "congestion", part_alg, brute_force, results=True)
#     assert solution.is_tree_partition()


@pytest.mark.parametrize("case, name, k, part_alg", cases())
def test_recursive(case, name, k, part_alg):
    c = case()
    solution = recursive(k, "congestion", part_alg, brute_force, results=True)
    assert solution


# def test_two_stage_milp(small_cases):
#     K = 5
#     objective = "congestion"
#     for _case in small_cases[:]:
#         case = _case()
#         for k in range(2, K + 1):
#             for part_alg in [normalized_modularity, normalized_laplacian, fastgreedy]:
#                 for ls_alg in [milp_line_switching]:
#                     solution = two_stage(k, objective, part_alg, ls_alg)
#                     assert solution.is_tree_partition()


# def test_two_stage_milp(medium_cases):
#     K = 5
#     objective = "congestion"
#     for _case in medium_cases[:]:
#         case = _case()
#         for k in range(2, K + 1):
#             for part_alg in [normalized_modularity, normalized_laplacian, fastgreedy]:
#                 for ls_alg in [milp_line_switching]:
#                     solution = two_stage(k, objective, part_alg, ls_alg)
#                     assert solution.is_tree_partition()


# def test_two_stage_small_brute_force(small_cases):
#     k = 2
#     objective = "congestion"
#     for _case in small_cases[:]:
#         case = _case()
#         for part_alg in [normalized_modularity, normalized_laplacian, fastgreedy]:
#             for ls_alg in [brute_force]:
#                 solution = two_stage(k, objective, part_alg, ls_alg)
#                 assert solution.is_tree_partition()


# def test_two_stage_milp_all(all_cases):
#     k = 2
#     objective = "congestion"
#     for _case in all_cases[:]:
#         case = _case()
#         for part_alg in [normalized_modularity, normalized_laplacian, fastgreedy]:
#             if part_alg(k).is_connected_clusters(case.G):
#                 for ls_alg in [milp_line_switching]:
#                     solution = two_stage(k, objective, part_alg, ls_alg)
#                     assert solution.is_tree_partition()
#                     print(case)
#             else:
#                 print(f"{case} not considered")
