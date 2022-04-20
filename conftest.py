import pytest
import os

from tree_partitioning.classes import Case


def make_callable_case(path, **kwargs):
    def f():
        return Case.from_file(path, **kwargs)

    return f


@pytest.fixture()
def ieee73(**kwargs):
    path = "data/pglib_opf_case73_ieee_rts.mat"
    return make_callable_case(path, **kwargs)


@pytest.fixture()
def small_cases(**kwargs):
    paths = [
        "data/pglib_opf_case14_ieee.mat",
        "data/pglib_opf_case24_ieee_rts.mat",
        "data/pglib_opf_case30_ieee.mat",
        "data/pglib_opf_case39_epri.mat",
        "data/pglib_opf_case57_ieee.mat",
        "data/pglib_opf_case73_ieee_rts.mat",
        "data/pglib_opf_case89_pegase.mat",
        "data/pglib_opf_case118_ieee.mat",
        "data/pglib_opf_case162_ieee_dtc.mat",
        "data/pglib_opf_case179_goc.mat",
    ]
    return [make_callable_case(path) for path in paths]


@pytest.fixture()
def medium_cases(**kwargs):
    paths = [
        "data/pglib_opf_case200_activ.mat",
        "data/pglib_opf_case240_pserc.mat",
        "data/pglib_opf_case300_ieee.mat",
        "data/pglib_opf_case500_goc.mat",
        "data/pglib_opf_case500_goc_postopf.mat",
        "data/pglib_opf_case588_sdet.mat",
        "data/pglib_opf_case793_goc.mat",
    ]
    return [make_callable_case(path) for path in paths]


@pytest.fixture()
def large_cases(**kwargs):
    paths = [
        "data/pglib_opf_case1354_pegase.mat",
        "data/pglib_opf_case1888_rte.mat",
        "data/pglib_opf_case1888_rte_postopf.mat",
        "data/pglib_opf_case1951_rte.mat",
        "data/pglib_opf_case2000_goc.mat",
        "data/pglib_opf_case2383wp_k.mat",
        "data/pglib_opf_case2383wp_k_postopf.mat",
        "data/pglib_opf_case2736sp_k.mat",
        "data/pglib_opf_case2737sop_k.mat",
        "data/pglib_opf_case2746wop_k.mat",
        "data/pglib_opf_case2746wp_k.mat",
        "data/pglib_opf_case2848_rte.mat",
        "data/pglib_opf_case2869_pegase.mat",
    ]
    return [make_callable_case(path) for path in paths]


@pytest.fixture()
def all_cases(**kwargs):
    paths = ["data/" + name for name in os.listdir("data/")]
    return [make_callable_case(path) for path in paths]
