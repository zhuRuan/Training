import pandas as pd
import numpy as np


'''
计算IC值
输入：
factor:因子值矩阵
ret:收益率矩阵
'''


def calculate_ic(factor: pd.DataFrame, ret: pd.DataFrame):
    factor.dropna(inplace=True)
    ret.dropna(inplace=True)
    factor = np.mat(factor.to_numpy())
    ret = np.mat(ret.to_numpy())
    a1_mean = np.average(factor,axis=1)
    a2_mean = np.average(ret,axis=1)

    a1 = (factor - a1_mean).A
    a2 = (ret - a2_mean).A
    list2 = []
    for row_r, row_f in zip(a2, a1):
        length = len(row_r) -1
        cov_pre = np.dot(row_r,row_f)
        cov = cov_pre / length
        list2.append(cov)
    var = np.multiply(factor.std(axis=1) ,ret.std(axis=1)).T.A
    ic = np.array(list2) / var

    return ic


def mono_dist(ret_list):
    # 计算加总
    ret_cum_list = []
    for series in ret_list:
        ret_cum_list.append(series.cumprod().tail(1))
    return ret_cum_list
