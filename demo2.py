# coding=utf-8
import os.path
import time

import numpy as np
import pandas as pd
import streamlit as st
from back_testing.back_testing2 import run_back_testing_new
from constant import trl_tuple
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import datetime

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    T1 = time.perf_counter()
    boxes = 10
    lamda = 0.2
    lag = 1
    rows = 200
    columns = 500
    np.random.seed(3)

    item_list = []

    for trl in trl_tuple:
        item_list.append((lamda, boxes, lag, rows, columns, trl))

    with ThreadPoolExecutor(max_workers=None) as executor:
        # 返回的
        res = executor.map(run_back_testing_new, item_list)

    dataframe_list = []
    for _elem in res:
        dataframe_list.extend(_elem)
    df = pd.concat(dataframe_list, )
    dir = 'pickle_data\\'

    if os.path.exists(dir + 'sum.csv'):
        df.to_csv(dir + 'sum.csv', header=0, mode='a', index=0)
    else:
        df.to_csv(dir + 'sum.csv', mode='a', index=0)
    T2 = time.perf_counter()
    print('本次用时:',T2-T1)