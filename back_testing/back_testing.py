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
    portfolio, ret_total, ret_list, ret_top, ret_short = computing(ret, dummy, CAP, lamda, boxes)

    # print("持仓矩阵：")
    # print(portfolio)

    # 净值曲线展示
    plot_return(total_return_matrix=ret_total.cumprod(),top_return_matrix=ret_top.cumprod(), bottom_return_matrix=ret_short.cumprod())

    # 因子暴露
    valid_number_matrix, dist_matrix, dist_mad_matrix = exposure(CAP)
    plot_exposure(valid_number_matrix=valid_number_matrix, dist_matrix=dist_matrix, dist_mad_matrix=dist_mad_matrix)

    # 单调性
    ic, ic_cum, _mono_dist = monotonicity(factor=CAP[dummy].iloc[:-lag, :],ret= ret[dummy].iloc[lag:, :], ret_df=ret_list)
    plot_monotonicity(mono_dist=_mono_dist, ic_list=ic, ic_cum_list=ic_cum)

