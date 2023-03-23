import pandas as pd
import numpy as np


'''
计算IC值
输入：
factor:因子值矩阵
ret:收益率矩阵
'''


def calculate_ic(factor: pd.DataFrame, ret: pd.DataFrame):
    _factor = factor.reset_index(drop=True)
    _ret = ret.reset_index(drop=True)
    factor_mean = _factor.mean(axis=1)
    ret_mean = _ret.mean(axis=1)

    a1 = (factor - factor_mean).values
    a2 = (ret - ret_mean).values
    list2 = []
    for _row in range(a1.shape[0]):
        cov = np.nanmean(a1[_row] * a2[_row])
        list2.append(cov)
    var = factor.std(axis=1).values * ret.std(axis=1).values
    ic = np.array(list2) / var

    return ic


def mono_dist(ret_list):
    # 计算加总
    ret_cum_list = []
    for series in ret_list:
        ret_cum_list.append(series.cumprod().tail(1))
    return ret_cum_list

def monotonicity(ret:pd.DataFrame, factor:pd.DataFrame, ret_list):
    ic = calculate_ic(ret, factor)
    ic_cum = ic.cumsum()
    _mono_dist = mono_dist(ret_list)
    return ic, ic_cum, _mono_dist
