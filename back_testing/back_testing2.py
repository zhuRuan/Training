# coding=utf-8
import time

import pandas as pd

from read_data.generate_random_data import *
from process_data.control import compute
from process_data.monotonicity import calculate_ic
import datetime
from constant import *
from read_data.get_data import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
import numpy as np
import traceback
import os


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return "Make dir success!"
    else:
        return "Failed."


def MaxDrawdown(return_list):
    '''最大回撤率'''
    matrix = return_list.copy(deep=True).reset_index(drop=True)
    i = np.argmax(
        (np.maximum.accumulate(matrix, axis=0) - matrix) / np.maximum.accumulate(matrix))  # 结束位置
    if i == 0:  # 等于0，说明没有回撤。
        return 0
    j = np.argmax(matrix[:i])  # 开始位置

    if i <= 0 and j <= 0:  # 小于0说明在组合中没有找到最大回撤点位，大概率是因为组合为空，原因不详
        return 0
    if not matrix.empty:
        num = (matrix[j] - matrix[i]) / matrix[j]
    else:
        num = 0
    return num


def MaxDrawdown_protfolio(return_matrix: pd.DataFrame):
    maxDrawdown_dict = {}
    maxDrawdown_list = []
    for column in list(return_matrix.columns):
        MaxDrawdown_num = MaxDrawdown(return_matrix[column])
        maxDrawdown_dict[column] = MaxDrawdown_num
        maxDrawdown_list.append(MaxDrawdown_num)
    return maxDrawdown_list


def annual_revenue(return_matrix: pd.DataFrame):
    '''
    计算年化收益率、夏普比率、最大回撤
    :param return_matrix: 收益率dataframe
    :return:
    '''
    std_list = return_matrix.std(axis=0).reset_index(drop=True)
    return_series = return_matrix.iloc[-1, :]
    annualized_rate_of_return = pd.Series(
        ((np.sign(return_series.values) * np.power(abs(return_series.values), 250 / len(return_matrix))) - 1).round(3))
    sharp_series = (annualized_rate_of_return / std_list).round(3)
    maximum_drawdown_series = pd.Series(MaxDrawdown_protfolio(return_matrix)).round(3)
    return annualized_rate_of_return.values, sharp_series.values, maximum_drawdown_series.values


def table_return(return_matrix: pd.DataFrame, ic_df: pd.DataFrame, method):
    '''生成三个部分的收益分析表格'''

    annual_ret, sharp, maximum_draw = annual_revenue(return_matrix=return_matrix)
    annual_ret_2, sharp_2, maximum_draw_2 = annual_revenue(
        return_matrix=return_matrix.iloc[:2 * int(len(return_matrix) / 3), :])
    annual_ret_3, sharp_3, maximum_draw_3 = annual_revenue(
        return_matrix=return_matrix.iloc[2 * int(len(return_matrix) / 3):, :])
    IC_mean = ic_df.mean(axis=0).round(3).iloc[0]
    ICIR = np.round(IC_mean / ic_df.std(axis=0).iloc[0], 3)
    table = pd.DataFrame({'因子名称': [factor_1, factor_1, factor_1], '使用的参数': [method, method, method],
                          '用作条件的因子': [factor_2, factor_2, factor_2],
                          '科目类别': list(return_matrix.columns)})
    table['nmlz_day'] = nmlz_day
    table['start_day'] = start_day
    table['end_day'] = end_day
    table['partion_loc'] = partition_loc
    table2 = pd.concat([table, pd.DataFrame({'年化收益率 （全时期）': annual_ret, '夏普比率 （全时期）': sharp,
                                             '最大回撤率 （全时期）': maximum_draw, '年化收益率 （前2/3时期）': annual_ret_2,
                                             '夏普比率 （前2/3时期）': sharp_2, '最大回撤率 （前2/3时期）': maximum_draw_2,
                                             '年化收益率 （后1/3时期）': annual_ret_3,
                                             '夏普比率 （后1/3时期）': sharp_3, '最大回撤率 （后1/3时期）': maximum_draw_3,
                                             'IC值': [IC_mean, IC_mean, IC_mean],
                                             'ICIR': [ICIR, ICIR, ICIR]})], axis=1)
    return table2


def detail_table(total_return_matrix, top_return_matrix, bottom_return_matrix, ic_df, method=''):
    return_matrix = pd.DataFrame([total_return_matrix, top_return_matrix, bottom_return_matrix]).T
    return_matrix.columns = ['LT_SB', "Long_top", "Long_bottom"]
    # 收益表格
    table = table_return(return_matrix, ic_df, method)
    return table, return_matrix


def get_matrices(rows, columns):
    # 生成三个矩阵，分别是收益率、成分股归属、市值
    ret = return_rate_matrix(rows, columns)
    dummy = dummy_matrix(rows, columns)
    CAP = CAP_matrix(rows, columns)
    return ret, dummy, CAP


def get_matrices2(rows, columns):
    # 生成三个矩阵，分别是收益率、成分股归属、市值
    ret = return_rate_matrix(rows, columns)
    dummy = dummy_matrix(rows, columns)
    CAP = CAP_matrix(rows, columns)
    VOL = volatility_matrix(rows, columns)
    return ret, dummy, CAP, VOL


def timing(matrix, _start, _end):
    '''
    選擇合適的時間
    :param matrix: 傳入的index為YYYYMMDD如20130605
    :param _start: 期望覆蓋的開始時間
    :param _end: 期望結束的開始時間
    :return: 返回的裁切好的矩陣
    '''
    matrix.index = pd.to_datetime(matrix.index)
    _return_matrix = (matrix.query('@_start<= index <= @_end'))
    return _return_matrix


# 生成四个矩阵(dataframe)：收益率，是否为指定成分股的dummy，市值， 波动率
def run_back_testing_new(x):
    try:
        T_read1 = time.perf_counter()
        # lamda = 0.2, boxes = 3, lag = 1, rows = 30, columns = 30, trl = 30
        plot_dict_dict = {}
        matrix_A_name = 'CAP'
        matrix_B_name = 'VOL'

        lamda, boxes, lag, rows, columns, trl = x

        # 获取数据矩阵，并鎖定時間區間
        start_date = datetime.datetime.strptime(start_day, '%Y%m%d')
        end_date = datetime.datetime.strptime(end_day, '%Y%m%d')
        ret = timing(get_ret_matrix(), start_date, end_date)
        dummy = timing(get_China_Securities_Index().replace(np.nan, False).replace(1.0, True), start_date, end_date)

        _factor_1 = timing(get_dv_ttm().shift(1), start_date, end_date)
        _factor_2 = timing(get_turnover_rate().shift(1), start_date, end_date)

        # 创建一个文件夹，用于装不同方法的数据
        dir = 'pickle_data\\' + str(start_date.strftime("%Y-%m")) + '_' + str(
            end_date.strftime("%Y-%m")) + '_' + factor_1 + '&&' + factor_2 + '_trl' + str(trl)
        mkdir(dir)

        # 用于装参数表格的list
        df_table_list = []
        T_read2 = time.perf_counter()
        print("读取数据用时：", T_read2 - T_read1)

        # 计算新因子，并计算其他变量
        for res in compute(ret, dummy, _factor_1, _factor_2, lamda, boxes, trl):  # 计算持仓矩阵和最终的收益率
            portfolio, ret_total, ret_boxes_df, ret_top, ret_bot, method, _factor_2_new, dummy_new, ret_new = res
            ic_df = calculate_ic(_factor_2_new, ret_new)

            T3 = time.perf_counter()
            # 参数汇总表
            detail_tab = detail_table(total_return_matrix=(ret_total + 1).cumprod() / (ret_total.iloc[0] + 1),
                                      top_return_matrix=(ret_top + 1).cumprod() / (ret_top.iloc[0] + 1),
                                      bottom_return_matrix=(ret_bot + 1).cumprod() / (ret_bot.iloc[0] + 1), ic_df=ic_df,
                                      method=method)

            # streamlit需要用的python变量打包
            plot_dict = {'ret_total': ret_total, 'ret_top': ret_top, 'ret_bot': ret_bot, 'ret_boxes_df': ret_boxes_df,
                         '_factor_2_new': _factor_2_new, 'dummy_new': dummy_new, 'ret_new': ret_new,
                         'factor_name1': matrix_A_name, 'factor_name2': matrix_B_name}
            plot_dict_dict[method] = plot_dict

            # pickle表格
            pickle_path = dir + '\\table_' + method + str('.csv')
            detail_tab[0].to_csv(pickle_path, index=False, encoding='gbk')

            df_table_list.append(detail_tab[0])
            T4 = time.perf_counter()
            print('生成表格时间为：', T4 - T3)

        # pickle表格
        with open(dir + '\\' + 'test.pkl', 'wb') as f:
            pickle.dump(plot_dict_dict, f, 0)
        return df_table_list

    except Exception as e:
        traceback.print_exc()