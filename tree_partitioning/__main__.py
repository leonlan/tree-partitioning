#!/usr/bin/env ipython
ieee5 = create("5_pjm", "IEEE-5", pn.case5())
ieee14 = create("14_ieee", "IEEE-14", pn.case14())
ieee24 = create("24_ieee_rts", "IEEE-24", None)
ieee30 = create("30_ieee", "IEEE-30", pn.case30())
epri39 = create("39_epri", "EPRI-39", pn.case39())
ieee57 = create("57_ieee", "IEEE-57", pn.case57())
ieee73 = create("73_ieee_rts", "IEEE-73", None)
ieee118 = create("118_ieee", "IEEE-118", pn.case118())
ieee162 = create("162_ieee_dtc", "IEEE-162", None)
ieee300 = create("300_ieee", "IEEE-300", pn.case300())
activ200 = create("200_activ", "ACTIV-200")

pserc240 = create("240_pserc", "PSERC-240")

goc179 = create("179_goc", "GOC-179")
goc500 = create("500_goc", "GOC-500")
goc793 = create("793_goc", "GOC-793")
goc2000 = create("2000_goc", "GOC-2000")

rte1888 = create("1888_rte", "RTE-1888", pn.case1888rte())
rte1951 = create("1951_rte", "RTE-1951")
pegase89 = create("89_pegase", "PEGASE-89")
pegase1354 = create("1354_pegase", "PEGASE-1354")
pegase2869 = create("2869_pegase", "PEGASE-2869")


sop2737 = create("2737sop_k", "SOP-2737")

all_cases = [
    ieee5,
    ieee14,
    ieee24,
    ieee30,
    ieee57,
    ieee73,
    ieee118,
    ieee300,
    epri39,
    activ200,
    pserc240,
    goc179,
    goc500,
    goc793,
    goc2000,
    rte1888,
    rte1951,
    pegase89,
    pegase1354,
    # pegase2869, # Length error
    sop2737,
]
