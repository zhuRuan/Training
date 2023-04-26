import pandas as pd

from read_data.generate_random_data import *
from process_data.return_rate import computing, multi_thread_cp2
# from plot_data.streamlit_plot import *
from process_data.exposure import exposure
from process_data.monotonicity import monotonicity
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from constant import calc_method
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
    matrix = return_list.copy().reset_index(drop=True)
    i = np.argmax(
        (np.maximum.accumulate(matrix, axis=0) - matrix) / np.maximum.accumulate(matrix))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(matrix[:i])  # 开始位置
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
    '''计算年化收益率、夏普比率、最大回撤'''
    std_list = return_matrix.std(axis=0)
    return_series = return_matrix.iloc[-1, :]
    annualized_rate_of_return = pd.Series(
        ((np.sign(return_series.values) * np.power(abs(return_series.values), 250 / len(return_matrix))) - 1).round(3))
    return_series = return_series - 1
    sharp_series = (return_series / std_list).round(3)
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
    return pd.DataFrame(
        {'因子名称': ['CAP', 'CAP', 'CAP'], '参数1': [method, method, method], '参数2': ['', '', ''],
         '科目类别': list(return_matrix.columns),
         '年化收益率 （全时期）': annual_ret, '夏普比率 （全时期）': sharp, '最大回撤率 （全时期）': maximum_draw, '年化收益率 （前2/3时期）': annual_ret_2,
         '夏普比率 （前2/3时期）': sharp_2, '最大回撤率 （前2/3时期）': maximum_draw_2, '年化收益率 （后1/3时期）': annual_ret_3,
         '夏普比率 （后1/3时期）': sharp_3, '最大回撤率 （后1/3时期）': maximum_draw_3, 'IC值': [IC_mean, IC_mean, IC_mean],
         'ICIR': [ICIR, ICIR, ICIR]})


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


# 生成三个矩阵(dataframe)：收益率，是否为指定成分股的dummy，最新市值
def run_back_testing(lamda=0.2, boxes=3, lag=1, rows=30, columns=30):
    ret, dummy, CAP = get_matrices(rows, columns, lag)

    # 数组运算
    portfolio, ret_total, ret_boxes_df, ret_top, ret_bot = computing(ret, dummy, CAP, lamda, boxes)

    # print("持仓矩阵：")
    # print(portfolio)

    # 因子暴露
    valid_number_matrix, dist_matrix, dist_mad_matrix = exposure(CAP)

    # 单调性
    lag_list = [1, 5, 20]
    ic = 0
    ic_cum_list = []
    mono_dist_list = []
    for _lag in lag_list:
        if _lag != 1:
            factor_matrix = CAP[dummy].iloc[:-(_lag - 1), :]
        else:
            factor_matrix = CAP[dummy].iloc[:, :]
        ret_matrix = (ret[dummy] + 1).rolling(_lag).apply(np.prod) - 1
        ret_boxes_matrix = (ret_boxes_df + 1).rolling(_lag).apply(np.prod) - 1
        _ic, _ic_cum, _mono_dist = monotonicity(factor=factor_matrix, ret=ret_matrix.iloc[(_lag - 1):, :],
                                                ret_df=ret_boxes_matrix)
        if _lag == 1:
            ic = _ic
        ic_cum_list.append(_ic_cum)
        mono_dist_list.append(_mono_dist)
    # # 净值曲线展示
    # plot_return(total_return_matrix=(ret_total + 1).cumprod(), top_return_matrix=(ret_top + 1).cumprod(),
    #             bottom_return_matrix=(ret_bot + 1).cumprod(), ic_df=ic)
    # # 因子暴露展示
    # plot_exposure(valid_number_matrix=valid_number_matrix, dist_matrix=dist_matrix, dist_mad_matrix=dist_mad_matrix)
    # # 单调性展示
    # plot_monotonicity(mono_dist=mono_dist_list, ic_list=ic, ic_cum_list=ic_cum_list, lag=_lag)


# 生成四个矩阵(dataframe)：收益率，是否为指定成分股的dummy，市值， 波动率
def run_back_testing_new(x):
    try:
        # lamda = 0.2, boxes = 3, lag = 1, rows = 30, columns = 30, trl = 30
        plot_dict_dict = {}
        matrix_A_name = 'CAP'
        matrix_B_name = 'VOL'
        # cacl_method = selectbox('您想要使用的方法是？', )
        lamda, boxes, lag, rows, columns, trl = x

        ret, dummy, CAP, VOL = get_matrices2(rows, columns)
        dir = 'pickle_data\\' + str(datetime.now().strftime("%Y-%m-%d_%H_%M")) + '_trl' + str(trl)
        mkdir(dir)
        # 数组运算
        for res in multi_thread_cp2(ret, dummy, CAP, VOL, lamda, boxes, trl):
            portfolio, ret_total, ret_boxes_df, ret_top, ret_bot, method = res

            # 因子暴露
            valid_number_matrix, dist_matrix, dist_mad_matrix = exposure(CAP)

            # 单调性
            lag_list = [1, 5, 20]
            ic = 0
            ic_cum_list = []
            mono_dist_list = []
            for _lag in lag_list:
                if _lag != 1:
                    factor_matrix = CAP[dummy].iloc[:-(_lag - 1), :]
                else:
                    factor_matrix = CAP[dummy].iloc[:, :]
                ret_matrix = (ret[dummy] + 1).rolling(_lag).apply(np.prod) - 1
                ret_boxes_matrix = (ret_boxes_df + 1).rolling(_lag).apply(np.prod) - 1
                _ic, _ic_cum, _mono_dist = monotonicity(factor=factor_matrix, ret=ret_matrix.iloc[(_lag - 1):, :],
                                                        ret_df=ret_boxes_matrix)
                if _lag == 1:
                    ic = _ic
                ic_cum_list.append(_ic_cum)
                mono_dist_list.append(_mono_dist)
            detail_tab = detail_table(total_return_matrix=(ret_total + 1).cumprod(),
                                      top_return_matrix=(ret_top + 1).cumprod(),
                                      bottom_return_matrix=(ret_bot + 1).cumprod(), ic_df=ic, method=method)

            plot_dict = {'ret_total': ret_total, 'ret_top': ret_top,
                         'ret_bot': ret_bot, 'ic_df': ic,
                         'valid_number_matrix': valid_number_matrix,
                         'dist_matrix': dist_matrix, 'dist_mad_matrix': dist_mad_matrix, 'mono_dist': mono_dist_list,
                         'ic_list': ic, 'ic_cum_list': ic_cum_list, 'lag': _lag, 'ret_matrix': detail_tab[1], 'factor_name1': matrix_A_name, 'factor_name2':matrix_B_name}
            plot_dict_dict[method] = plot_dict
            # pickle表格

            pickle_path = dir + '\\table_' + method + str('.csv')
            detail_tab[0].to_csv(pickle_path, index=False, encoding='gbk')
        # pickle表格
        with open(dir + '\\' + 'test.pkl', 'wb') as f:
            pickle.dump(plot_dict_dict, f, 0)
        print(plot_dict_dict)
        # if method == cacl_method:
        #     # 净值曲线展示
        #     plot_return(total_return_matrix=(ret_total + 1).cumprod(), top_return_matrix=(ret_top + 1).cumprod(),
        #                 bottom_return_matrix=(ret_bot + 1).cumprod(), ic_df=ic)
        #     # 因子暴露展示
        #     plot_exposure(valid_number_matrix=valid_number_matrix, dist_matrix=dist_matrix, dist_mad_matrix=dist_mad_matrix)
        #     # 单调性展示
        #     plot_monotonicity(mono_dist=mono_dist_list, ic_list=ic, ic_cum_list=ic_cum_list, lag=_lag)
    except Exception as e:
        traceback.print_exc()
