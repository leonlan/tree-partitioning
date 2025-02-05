def format_case_name(name):
    """
    Returns formatted case names.
    """
    for key, value in case2name.items():
        if key in name:
            return value
    return name


case2name = {
    "5_pjm": "IEEE-5",
    "14_ieee": "IEEE-14",
    "24_ieee_rts": "IEEE-24",
    "30_ieee": "IEEE-30",
    "39_epri": "EPRI-39",
    "57_ieee": "IEEE-57",
    "73_ieee_rts": "IEEE-73",
    "89_pegase": "PEGASE-89",
    "118_ieee": "IEEE-118",
    "162_ieee_dtc": "IEEE-162",
    "179_goc": "GOC-179",
    "200_activ": "ACTIV-200",
    "240_pserc": "PSERC-240",
    "300_ieee": "IEEE-300",
    "500_goc": "GOC-500",
    "588_sdet": "SDET-588",
    "793_goc": "GOC-793",
    "1354_pegase": "PEGASE-1354",
    "1888_rte": "RTE-1888",
    "1951_rte": "RTE-1951",
    "2000_goc": "GOC-2000",
    "2869_pegase": "PEGASE-2869",
    "2737sop": "SOP-2737",
    "2736sp": "SP-2736",
    "2746wp_k": "WP-2746",
    "2746wop_k": "WOP-2746",
    "2848_rte": "RTE-2848",
}


# For power flow disruption and maximum congestion
results_columns = [
    "case",
    "n_buses",
    "n_lines",
    "n_clusters",
    "pre_congestion",
    "algorithm",
    "power_flow_disruption",
    "maximum_congestion",
    "runtime",
    "n_cross_edges",
    "n_switched_lines",
    "cluster_sizes",
    "generator_sizes",
    "generators",
    "partition",
    "lines",
    "mip_gap_single_stage",
    "mip_gap_partitioning_stage",
    "mip_gap_line_switching_stage",
]
