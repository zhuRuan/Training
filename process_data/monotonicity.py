import pandas as pd
import numpy as np


def calculate_ic(factor: pd.DataFrame, ret: pd.DataFrame):
    '''
    计算IC值
    输入：
    factor:因子值矩阵
    ret:收益率矩阵
    '''
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

    # 将ic从series变为dataframe
    ic_df = pd.DataFrame(ic)
    ic_df.columns = ['IC_CAP']
    return ic_df


def mono_dist(ret_df):
    # 计算加总
    ret_cum_df = (ret_df + 1).cumprod().iloc[-1]
    ret_cum_df = ret_cum_df.to_frame()
    ret_cum_df['boxes'] = ret_cum_df.index
    ret_cum_df.columns = ['return_rate', 'boxes']

    return ret_cum_df


def monotonicity(ret: pd.DataFrame, factor: pd.DataFrame, ret_df):
    ic = calculate_ic(ret, factor)
    ic_cum = ic.cumsum()
    ic_cum.columns = ['IC_CUM_CAP']
    _mono_dist = mono_dist(ret_df)
    return ic, ic_cum, _mono_dist
