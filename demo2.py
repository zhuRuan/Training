# coding=utf-8
import os.path
import time

import numpy as np
import pandas as pd
from back_testing.back_testing2 import run_back_testing_new
from constant import trl_tuple, nmlz_days_tuple, start_date, end_date, factor_1, factor_2, partition_loc
from tqdm import tqdm



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    T1 = time.perf_counter()

    res_list = run_back_testing_new()
    df = pd.concat(res_list, axis=0)

    # 存储sum.csv
    dir = 'pickle_data\\'
    if os.path.exists(dir + 'sum.csv'):
        df.to_csv(dir + 'sum.csv', header=0, mode='a', index=0)
    else:
        df.to_csv(dir + 'sum.csv', mode='a', index=0)


    T2 = time.perf_counter()
    print('本次用时:',T2-T1)