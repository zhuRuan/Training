from read_data.generate_random_data import *
from process_data.return_rate import computing
from plot_data.winsorize_plot import winsorize_plot
from process_data.exposure import exposure
from process_data.monotonicity import mono_dist, calculate_ic
import matplotlib.pyplot as plt
import numpy as np


def get_matrices():
    # 生成三个矩阵，分别是收益率、成分股归属、市值
    ret = return_rate_matrix()
    dummy = dummy_matrix()
    CAP = CAP_matrix()
    return ret, dummy, CAP


# 生成三个矩阵(dataframe)：收益率，是否为指定成分股的dummy，最新市值
def run_back_testing(lamda=0.2, boxes=3):
    ret, dummy, CAP = get_matrices()

    # 数组运算
    portfolio, ret_total, ret_list, ret_top, ret_short = computing(ret, dummy, CAP, lamda, boxes)

    print("持仓矩阵：")
    print(portfolio)

    # 净值曲线展示
    ret_cum = ret_total.cumprod()
    ret_cum.plot()
    plt.show()

    # 因子暴露
    exposure(CAP)

    # 单调性
    ic = calculate_ic(CAP[dummy].loc[:len(CAP) - 2, :], ret[dummy].loc[1:, :])
    ic_cum = ic.cumsum()
    mono_dist(ret_list)
