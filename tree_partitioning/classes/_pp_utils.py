import numpy as np
import pandapower as pp
import pandapower.converter as pc
import pandas as pd
from numpy.testing import assert_almost_equal


def _load_pp_case(path: str, opf_init: bool, ac: bool):
    """
    Loads the selected test case.

    - opf_init (bool): Run an initial OPF instead of regular PF
    - ac (bool): Runs on AC power flows
    """
    net = pc.from_mpc(path)

    # Run (O)PF or change ref bus first
    if ac:
        try:
            pp.runopp(net) if opf_init else pp.runpp(net)
        except UserWarning:
            ref_gen = net.gen.bus.iloc[0]  # bus index of first generator
            _change_ref_bus(net, ref_gen, ext_grid_p=0)
            pp.runopp(net) if opf_init else pp.runpp(net)
    else:
        try:
            pp.rundcopp(net) if opf_init else pp.rundcpp(net)
        except UserWarning:
            ref_gen = net.gen.bus.iloc[0]  # bus index of first generator
            _change_ref_bus(net, ref_gen, ext_grid_p=0)
            pp.rundcopp(net) if opf_init else pp.rundcpp(net)

    return net


def _netdict_from_pp_net(net, merge_lines):
    """
    - merge_lines (bool): Returns merged network.
    """
    dict_line = {i: i for i in range(len(net.line))}
    dict_trafo = {i: len(net.line) + i for i in range(len(net.trafo))}

    # Change load/generator power injections setpoints according to OPF
    net.load["p_mw"] = net.res_load["p_mw"]
    net.gen["p_mw"] = net.res_gen["p_mw"]

    # Change edge names to L#/T#
    net.line["name"] = [f"L{idx}" for idx in net.line.index]
    net.trafo["name"] = [f"T{idx}" for idx in net.trafo.index]

    # Create df of the network (lines)
    dfnetwork = net.line[["from_bus", "to_bus", "in_service", "name"]].rename(
        index=dict_line
    )
    dfnetwork = dfnetwork.combine_first(
        net.trafo[["hv_bus", "lv_bus", "in_service", "name"]].rename(
            columns={"hv_bus": "from_bus", "lv_bus": "to_bus"}, index=dict_trafo
        )
    )
    dfnetwork["from_bus"] = dfnetwork["from_bus"].astype(int)
    dfnetwork["to_bus"] = dfnetwork["to_bus"].astype(int)
    dfnetwork["weight"] = abs(
        _combine_line_trafo(net, dict_line, dict_trafo, "p_from_mw", "p_hv_mw")
    )
    dfnetwork["f"] = abs(
        _combine_line_trafo(net, dict_line, dict_trafo, "p_from_mw", "p_hv_mw")
    )
    dfnetwork["loading_percent"] = _combine_line_trafo(
        net, dict_line, dict_trafo, "loading_percent"
    )
    dfnetwork["type"] = dfnetwork["name"].apply(lambda x: x[0])
    dfnetwork["index_by_type"] = dfnetwork["name"].apply(lambda x: int(x[1:]))
    dfnetwork["edge_index"] = dfnetwork.index
    dfnetwork["b"] = _compute_susceptances(net, dfnetwork)
    dfnetwork["c"] = _compute_capacities(net, dfnetwork)
    dfnetwork["edge_id"] = tuple(zip(dfnetwork["from_bus"], dfnetwork["to_bus"]))
    dfnetwork.drop(dfnetwork[dfnetwork["in_service"] == False].index, inplace=True)

    # Create df of the buses
    df_bus = net.res_bus
    df_bus["p_mw"] = df_bus["p_mw"]  # positive is net consumption
    df_bus.loc[net.gen.bus, "p_gen"] = net.res_gen["p_mw"].values
    df_bus.loc[net.gen.bus, "min_p_mw"] = net.gen["min_p_mw"].values
    df_bus.loc[net.gen.bus, "max_p_mw"] = net.gen["max_p_mw"].values
    df_bus.loc[net.load.bus, "p_load"] = net.res_load["p_mw"].values

    # External grid: positive is net consumption
    df_bus.loc[net.ext_grid.bus, "p_ext_grid"] = net.res_ext_grid["p_mw"].values

    if not net.sgen.empty:  # has static generators
        # There could be multiple generators to one bus
        sgen_bus_idx = net.sgen.bus.unique()
        temp_df_bus = pd.concat([net.sgen["bus"], net.res_sgen["p_mw"]], axis=1)
        p_sgen = temp_df_bus.groupby(["bus"]).sum()["p_mw"]
        df_bus.loc[sgen_bus_idx, "p_sgen"] = p_sgen
    else:
        df_bus["p_sgen"] = 0

    if not net.shunt.empty:
        df_bus.loc[net.shunt.bus, "p_shunt"] = net.res_shunt["p_mw"].values
    else:
        df_bus["p_shunt"] = 0

    df_bus = df_bus.fillna(0)  # unfilled entries

    # TODO see issue https://github.com/leonlan/tree-partitioning/issues/4
    df_bus = df_bus.round(5)

    # The total generation and load is useful for cascading failures, where we
    # need fine grained control over the generation and/or load to do load shedding
    df_bus["p_gen_total"] = (
        df_bus["p_gen"] + df_bus["p_sgen"] + np.maximum(df_bus["p_ext_grid"], 0)
    )
    df_bus["p_load_total"] = (
        df_bus["p_load"] + df_bus["p_shunt"] + np.maximum(-df_bus["p_ext_grid"], 0)
    )

    # Check that p_mw is the same as the total load minus total generation
    assert_almost_equal(
        df_bus["p_mw"].values, (df_bus["p_load_total"] - df_bus["p_gen_total"]).values
    )

    # Consider merging lines
    if merge_lines:
        dfnetwork = (
            dfnetwork.groupby(["from_bus", "to_bus"])
            .agg(
                {
                    "in_service": lambda x: all(list(x)),
                    "name": lambda x: list(x),
                    "f": "sum",
                    "weight": "sum",
                    "loading_percent": "sum",
                    "edge_index": lambda x: min(list(x)),
                    "type": lambda x: list(x),
                    "index_by_type": lambda x: list(x),
                    "b": "sum",
                    "c": "sum",
                    "edge_id": lambda x: list(x),
                }
            )
            .reset_index()
        )
        dfnetwork["loading_percent"] = dfnetwork["weight"] / dfnetwork["c"] * 100

    dfnetwork.sort_values(by=["edge_index"], inplace=True)

    lines = {
        (data["from_bus"], data["to_bus"], i): data
        for i, (line, data) in enumerate(dfnetwork.T.to_dict().items())
    }
    netdict = {"buses": df_bus.T.to_dict(), "lines": lines}

    return netdict


"""
Helper functions
"""


def _combine_line_trafo(net, dict_line, dict_trafo, cols1, cols2=None):
    """Combines the results of line and transformer dataframes."""
    if not cols2:
        cols2 = cols1
    res = net.res_line[cols1].rename(index=dict_line)
    res = res.combine_first(net.res_trafo[cols2].rename(index=dict_trafo))
    return res


def _compute_susceptances(net, df):
    """
    Compute susceptances of the network.
    - net: pandapower network
    - df: dataframe of the network
    """
    # Line values
    b_lines = (
        np.array(
            1
            / (
                net.line["x_ohm_per_km"]
                * net.line["length_km"]
                * net.sn_mva
                / net.line["parallel"]
            )
        )
        * net.bus.loc[net.line.from_bus.values, "vn_kv"].values ** 2
    )

    # Transformer susceptances
    zk = net.trafo["vk_percent"] / 100 * net.sn_mva / net.trafo["sn_mva"]
    rk = net.trafo["vkr_percent"] / 100 * net.sn_mva / net.trafo["sn_mva"]
    xk = np.array(zk * zk - rk * rk) ** (1 / 2)

    # Fill nans in tap_step_percent with 0
    net_trafo_tap_step_percent = net.trafo["tap_step_percent"].fillna(0)
    tapratiok = np.array(1 - net_trafo_tap_step_percent / 100)
    b_trafo = 1 / (xk * tapratiok)

    b = np.append(b_lines, b_trafo)

    return b


def _change_ref_bus(net, ref_bus_idx, ext_grid_p=0):  # Copied from pandapower
    """
    This function changes the current reference bus / buses, declared by
    net.ext_grid.bus towards the given 'ref_bus_idx'. If ext_grid_p is a list,
    it must be in the same order as net.ext_grid.index.
    """
    # Cast ref_bus_idx and ext_grid_p as list
    if not isinstance(ref_bus_idx, list):
        ref_bus_idx = [ref_bus_idx]
    if not isinstance(ext_grid_p, list):
        ext_grid_p = [ext_grid_p]
    for i in ref_bus_idx:
        if i not in net.gen.bus.values and i not in net.ext_grid.bus.values:
            raise ValueError("Index %i is not in net.gen.bus or net.ext_grid.bus." % i)

    # Determine indices of ext_grid and gen connected to ref_bus_idx
    gen_idx = net.gen.index[net.gen.bus.isin(ref_bus_idx)]
    ext_grid_idx = net.ext_grid.index[~net.ext_grid.bus.isin(ref_bus_idx)]
    # old ext_grid -> gen
    j = 0
    for i in ext_grid_idx:
        ext_grid_data = net.ext_grid.loc[i]
        net.ext_grid.drop(i, inplace=True)
        pp.create_gen(
            net,
            ext_grid_data.bus,
            ext_grid_p[j],
            vm_pu=ext_grid_data.vm_pu,
            controllable=True,
            min_q_mvar=ext_grid_data.min_q_mvar,
            max_q_mvar=ext_grid_data.max_q_mvar,
            min_p_mw=ext_grid_data.min_p_mw,
            max_p_mw=ext_grid_data.max_p_mw,
        )
        j += 1
    # old gen at ref_bus -> ext_grid (and sgen)
    for i in gen_idx:
        gen_data = net.gen.loc[i]
        net.gen.drop(i, inplace=True)
        if gen_data.bus not in net.ext_grid.bus.values:
            pp.create_ext_grid(
                net,
                gen_data.bus,
                vm_pu=gen_data.vm_pu,
                va_degree=0.0,
                min_q_mvar=gen_data.min_q_mvar,
                max_q_mvar=gen_data.max_q_mvar,
                min_p_mw=gen_data.min_p_mw,
                max_p_mw=gen_data.max_p_mw,
            )
        else:
            pp.create_sgen(
                net,
                gen_data.bus,
                p_mw=gen_data.p_mw,
                min_q_mvar=gen_data.min_q_mvar,
                max_q_mvar=gen_data.max_q_mvar,
                min_p_mw=gen_data.min_p_mw,
                max_p_mw=gen_data.max_p_mw,
            )


def _compute_capacities(net, df):
    """Compute line and trafo capacities."""
    c_line = (
        net.line.max_i_ka
        * net.bus.loc[net.line.from_bus.values, "vn_kv"].values
        / (np.sqrt(3) / 3)
    )
    c_trafo = net.trafo.sn_mva.values
    c = np.append(c_line, c_trafo)
    return np.array(c, dtype="int")
