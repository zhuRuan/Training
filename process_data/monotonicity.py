import pandas as pd
import numpy as np


'''
计算IC值
输入：
factor:因子值矩阵
ret:收益率矩阵
'''


def calculate_ic(factor: pd.DataFrame, ret: pd.DataFrame):
    ret.index = factor.index
    factor_mean = factor.mean(axis=1)
    ret_mean = ret.mean(axis=1)
    a1 = (factor - factor_mean).fillna(value=0)
    a2 = (ret - ret_mean).fillna(value=0)
    a3 = a2.transpose()
    matrix = np.dot(a1, a3)
    numbers = factor.count(axis=1)
    cov = np.diagonal(matrix) / (numbers - 1)
    std_factor = factor.std(axis=1)
    std_ret = ret.std(axis=1)

    ic = cov / (std_factor * std_ret)

    return ic


def mono_dist(ret_list):
    # 计算加总
    ret_cum_list = []
    for series in ret_list:
        ret_cum_list.append(series.cumprod().tail(1))
    return ret_cum_list
