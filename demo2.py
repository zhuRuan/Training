# coding=utf-8
import os.path
import time
import traceback

import numpy as np
import pandas as pd
from back_testing.back_testing2 import run_back_testing_new
from constant import factor_1_name_list, factor_2_name_list, save_path, calc_method_tuple, nmlz_days_tuple
from tqdm import tqdm

pd.set_option('display.max_rows', 50)
pd.set_option('expand_frame_repr', False)


def save_csv(df):
    # 存储sum.csv
    dir = save_path + '\\'
    if os.path.exists(dir + 'sum.csv'):
        df.to_csv(dir + 'sum.csv', header=0, mode='a', index=0)
    else:
        df.to_csv(dir + 'sum.csv', mode='a', index=0)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    wrong_list = []  # 错误列表

    # 读取sum.csv列表
    if os.path.exists(save_path + '\\sum.csv'):
        sum_csv = pd.read_csv(save_path + '\\sum.csv')
    factor_pair_list = []

    # 添加要回测的因子
    for factor_1 in (factor_1_name_list):
        for factor_2 in factor_2_name_list:
            for calc_method in calc_method_tuple:
                for nmlz_days in nmlz_days_tuple:
                    # 判断csv文件是否存在，不存在，直接添加所有的因子组合，若存在则判断是否已经在csv列表内。
                    if not os.path.exists(save_path + '\\sum.csv') or (
                            (factor_1, factor_2, calc_method, nmlz_days) not in zip(sum_csv['因子名称'].values.tolist(),
                                                                                    sum_csv['用作条件的因子'].values.tolist(),
                                                                                    sum_csv['使用的参数'].values.tolist(),
                                                                                    sum_csv['nmlz_days'].values.tolist()
                                                                                    )):
                        factor_pair_list.append((factor_1, factor_2, calc_method, nmlz_days))

    # 开始带进度条的因子回测
    with tqdm(total=len(factor_pair_list)) as pbar:
        # 设置回测进度条的前缀说明
        pbar.set_description('因子回测运行中：')
        for pair in factor_pair_list:
            # 读取列表中的因子
            factor_1 = pair[0]
            factor_2 = pair[1]
            calc_method = pair[2]
            nmlz_days = pair[3]
            pbar.set_postfix({'factor_1': factor_1, 'factor_2': factor_2, 'calc_method': calc_method,
                              'nmlz_days': nmlz_days})
            back_testing_return = run_back_testing_new(factor_1, factor_2, calc_method, nmlz_days)

            # 判断返回的值，并处理文件
            if back_testing_return != None:  # 若符合条件，则保存csv
                df = pd.concat(back_testing_return, axis=0)
                save_csv(df)
            else:  # 不符合条件，则将出错的因子组合添加进错误列表，并打印错误列表
                wrong_list.append((factor_1, factor_2))
                print(wrong_list)

            # 更新pbar进度
            pbar.update(1)
