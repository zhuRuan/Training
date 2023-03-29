import pandas as pd

from read_data.generate_random_data import *
from process_data.return_rate import computing
from plot_data.streamlit_plot import *
from process_data.exposure import exposure
from process_data.monotonicity import monotonicity
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_resource
def get_matrices(rows, columns, lag: int):
    # 生成三个矩阵，分别是收益率、成分股归属、市值
    ret = return_rate_matrix(rows, columns)
    dummy = dummy_matrix(rows, columns)
    CAP = CAP_matrix(rows, columns)
    return ret, dummy, CAP


# 生成三个矩阵(dataframe)：收益率，是否为指定成分股的dummy，最新市值
def run_back_testing(lamda=0.2, boxes=3, lag=1, rows=30, columns=30):
    ret, dummy, CAP = get_matrices(rows, columns, lag)

    # 数组运算
    portfolio, ret_total, ret_list, ret_top, ret_bot = computing(ret, dummy, CAP, lamda, boxes)

    # print("持仓矩阵：")
    # print(portfolio)

    # 净值曲线展示
    plot_return(total_return_matrix=(ret_total + 1).cumprod(), top_return_matrix=(ret_top + 1).cumprod(),
                bottom_return_matrix=(ret_bot + 1).cumprod())

    # 因子暴露
    valid_number_matrix, dist_matrix, dist_mad_matrix = exposure(CAP)
    plot_exposure(valid_number_matrix=valid_number_matrix, dist_matrix=dist_matrix, dist_mad_matrix=dist_mad_matrix)

    # 单调性
    lag_list = [1,5,20]
    ic = 0
    ic_cum_list = []
    mono_dist_list = []
    for _lag in lag_list :
        factor_matrix = CAP[dummy].iloc[:-_lag, :]
        ret_matrix = (ret[dummy]+1).rolling(lag).apply(np.prod) -1
        _ic, _ic_cum, _mono_dist = monotonicity(factor=factor_matrix, ret=ret_matrix.iloc[_lag:, :],
                                              ret_df=ret_list)
        if _lag == 1:
            ic = _ic
        ic_cum_list.append(_ic_cum)
        mono_dist_list.append(_mono_dist)
    plot_monotonicity(mono_dist=mono_dist_list, ic_list=ic, ic_cum_list=ic_cum_list, lag=_lag)
