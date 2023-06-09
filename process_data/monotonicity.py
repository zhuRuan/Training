import pandas
import pandas as pd
import numpy as np


def calculate_ic(factor: pd.DataFrame(), ret: pd.DataFrame()):
    '''
    计算IC值
    输入：
    factor:因子值矩阵
    ret:收益率矩阵
    '''
    _factor = factor.copy(deep=True)
    _factor = _factor.reset_index(drop=True) # 同步坐标，否则会出现问题
    _ret = ret.copy(deep=True)
    _ret = _ret.reset_index(drop=True)

    a1 = (_factor.sub(_factor.mean(axis=1),axis=0))
    a2 = (_ret.sub(_ret.mean(axis=1),axis=0))
    ic = (a1 * a2).mean(axis=1)/(_factor.std(axis=1)+ 1e-8)/(_ret.std(axis=1) + 1e-8)

    # 将ic从series变为dataframe
    ic_df = pd.DataFrame(ic)
    ic_df.columns = ['IC']
    return ic_df


def mono_dist(ret_df: pandas.DataFrame):
    # 计算加总
    ret_cum_df = ret_df.iloc[-1]
    ret_cum_df = ret_cum_df.to_frame()
    ret_cum_df['boxes'] = ret_cum_df.index
    ret_cum_df.columns = ['return_rate_minus_mean', 'boxes']
    ret_cum_df['return_rate_minus_mean'] = ret_cum_df['return_rate_minus_mean'] - ret_cum_df['return_rate_minus_mean'].mean()

    return ret_cum_df


def monotonicity(ret: pd.DataFrame, factor: pd.DataFrame, ret_df):
    ic = calculate_ic(ret, factor)
    ic_cum = ic.cumsum()
    ic_cum.columns = ['IC_CUM_CAP']
    _mono_dist = mono_dist(ret_df)
    return ic, ic_cum, _mono_dist

