import pandas as pd
import numpy as np
import datetime
import time

from read_data.get_data import get_total_mv, get_volumn_ratio, get_ret_matrix
from back_testing.back_testing2 import timing
from constant import start_day, end_day, top_ratio, trl_tuple, partition_loc


def calculate_ic(factor: pd.DataFrame(), ret: pd.DataFrame()):
    '''
    计算IC值
    输入：
    factor:因子值矩阵
    ret:收益率矩阵
    '''
    _factor = factor.copy(deep=True)
    _factor = _factor.reset_index(drop=True)  # 同步坐标，否则会出现问题
    _ret = ret.copy(deep=True)
    _ret = _ret.reset_index(drop=True)

    a1 = (_factor.sub(_factor.mean(axis=1), axis=0))
    a2 = (_ret.sub(_ret.mean(axis=1), axis=0))
    ic = (a1 * a2).mean(axis=1) / _factor.std(axis=1) / _ret.std(axis=1)

    # 将ic从series变为dataframe
    ic_df = pd.DataFrame(ic)
    ic_df.columns = ['IC']
    return ic_df


start_date = datetime.datetime.strptime(start_day, '%Y%m%d')
end_date = datetime.datetime.strptime(end_day, '%Y%m%d')
_factor_1 = timing(get_total_mv('../is/basic/total_mv.pkl').shift(1), start_date, end_date)
_factor_2 = timing(get_volumn_ratio('../is/basic/volume_ratio.pkl').shift(1), start_date, end_date)

T1 = time.perf_counter()
c = calculate_ic(_factor_2, _factor_1)
T2 = time.perf_counter()
print('time:', T2 - T1)
a = pd.DataFrame([[1, 2, 3, 4], [3, 4, 5, 56], [3, 4, 5, 6, 7], [4, 4, 4, 4, 4]])
b = pd.DataFrame([[np.nan, 2, 3, 4], [np.nan, np.nan, np.nan, 3]])
print(a.rolling(window=2, axis=0).apply(np.prod))
