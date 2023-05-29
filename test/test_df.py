# coding=utf-8
import numpy as np
import pandas as pd
import datetime
import time
from read_data.get_data import get_total_mv, get_volumn_ratio
from back_testing.back_testing2 import timing
from constant import start_day, end_day, top_ratio, trl_tuple, partition_loc

k = 0


def group_rolling(trl, factor_1, factor_2):
    '''
    进行滚动
    :param n: 滚动的行数
    :param df: 目标数据框
    :return:
    '''
    list1 = []
    for i in range(len(factor_1) - trl + 1):
        rank_df = factor_1.iloc[i:i + trl, :].rank(axis=0, method='dense', ascending=False, na_option='keep', pct=True)
        if partition_loc == 'TOP':
            _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
        else:
            _true_false_df = rank_df >= 1 - top_ratio
        _true_false_df = _true_false_df.replace(False, -1)
        _result_df = factor_2.iloc[k:k + trl, :] * _true_false_df
        _result_df[_result_df<0] = np.nan
        list1.append(_result_df.mean(axis=0))
    return pd.concat(list1, axis=0)


def group_rolling2_mean(df, factor_1, factor_2, trl, max_k):
    global k
    rank_df = factor_1.iloc[k:k + trl, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    if partition_loc == 'TOP':
        _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
    else:
        _true_false_df = rank_df >= 1- top_ratio
    _result_df = factor_2.iloc[k:k + trl, :][_true_false_df]
    k = k + 1
    if max_k - 1 > k:
        return _result_df.mean(axis=0)
    else:
        k = 0

def group_rolling2_mean2(df, factor_1, factor_2, trl, max_k):
    global k
    rank_df = factor_1.iloc[k:k + trl, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    # if partition_loc == 'TOP':
    _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
    # else:
    #     _true_false_df = rank_df >= 1- top_ratio
    _result_df = factor_2.iloc[k:k + trl, :][_true_false_df]
    k = k + 1
    if max_k - 1 > k:
        return _result_df.mean(axis=0)
    else:
        k = 0


start_date = datetime.datetime.strptime(start_day, '%Y%m%d')
end_date = datetime.datetime.strptime(end_day, '%Y%m%d')
_factor_1 = timing(get_total_mv('../is/basic/total_mv.pkl').shift(1), start_date, end_date)
_factor_2 = timing(get_volumn_ratio('../is/basic/volume_ratio.pkl').shift(1), start_date, end_date)

list1 = []
T1 = time.perf_counter()
n = _factor_1.apply(group_rolling2_mean2, factor_1=_factor_1, factor_2=_factor_2, trl=trl_tuple[0], max_k=len(_factor_1), axis=1)
T2 = time.perf_counter()
list1.append(T2-T1)
print('程序运行时间：%s毫秒' % ((T2 - T1) * 1000))

T1 = time.perf_counter()
n = _factor_1.apply(group_rolling2_mean, factor_1=_factor_1, factor_2=_factor_2, trl=trl_tuple[0],
                    max_k=len(_factor_1), axis=1)
T2 = time.perf_counter()
print('程序运行时间：%s毫秒' % ((T2 - T1) * 1000))

T3 = time.perf_counter()
n2 = group_rolling(trl=2, factor_1=_factor_1, factor_2=_factor_2)
T4 = time.perf_counter()
print('程序运行时间：%s毫秒' % ((T4 - T3) * 1000))
print(n)
print(n2)
