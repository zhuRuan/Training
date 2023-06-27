# coding=utf-8
import time
import warnings

import pandas as pd
from empyrical import max_drawdown, sharpe_ratio, cum_returns, annual_return
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


def annual_revenue_total(return_matrix: pd.DataFrame):
    '''
    计算年化收益率、夏普比率、最大回撤
    :param return_matrix: 收益率dataframe
    :return:
    '''
    # 求出年化收益
    annualized_rate_of_return_series = annual_return(return_matrix.iloc[:, :3])
    # 将收益率变为涨跌了多少而非净值的多少
    sharp_series = pd.to_numeric(pd.Series(sharpe_ratio(return_matrix.iloc[:, :3])))
    # 求最大回撤
    maximum_drawdown_series = pd.Series(max_drawdown(return_matrix.iloc[:, :3]))
    # 求超额收益
    excess_return = annualized_rate_of_return_series - annual_return(return_matrix.iloc[:, 3])
    return annualized_rate_of_return_series.apply(lambda x: format(x, '.2%')).values, sharp_series.apply(
        lambda x: format(x, '.2f')).values, maximum_drawdown_series.apply(
        lambda x: format(x, '.2%')).values, excess_return.apply(lambda x: format(x, '.2%')).values


def annual_revenue(return_matrix: pd.DataFrame):
    '''
    计算年化收益率、夏普比率、最大回撤
    :param return_matrix: 收益率dataframe
    :return:
    '''
    # 求出年化收益
    annualized_rate_of_return_series = annual_return(return_matrix.iloc[:, :3])
    # 将收益率变为涨跌了多少而非净值的多少
    sharp_series = pd.to_numeric(pd.Series(sharpe_ratio(return_matrix.iloc[:, :3])))
    # 求最大回撤
    maximum_drawdown_series = pd.Series(max_drawdown(return_matrix.iloc[:, :3]))

    return annualized_rate_of_return_series.apply(lambda x: format(x, '.2%')).values, sharp_series.apply(
        lambda x: format(x, '.2f')).values, maximum_drawdown_series.apply(lambda x: format(x, '.2%')).values


def table_return(return_matrix: pd.DataFrame, ic_df: pd.DataFrame, method, trl, nmlz_days, sector_member, factor_1_name,
                 factor_2_name):
    '''生成三个部分的收益分析表格'''

    annual_ret, sharp, maximum_draw, excess_return = annual_revenue_total(return_matrix=return_matrix)
    annual_ret_2, sharp_2, maximum_draw_2 = annual_revenue(
        return_matrix=return_matrix.iloc[:2 * int(len(return_matrix) / 3)])
    annual_ret_3, sharp_3, maximum_draw_3 = annual_revenue(
        return_matrix=return_matrix.iloc[2 * int(len(return_matrix) / 3):])

    IC_mean = ic_df.mean(axis=0).round(3).iloc[0]
    ICIR = np.round(IC_mean / ic_df.std(axis=0).iloc[0], 3)
    table = pd.DataFrame({'因子名称': [factor_1_name, factor_1_name, factor_1_name], '用作条件的因子': [factor_2_name, factor_2_name, factor_2_name],
                          '使用的参数': [method, method, method], '归属指数': [sector_member, sector_member, sector_member],
                          '科目类别': list(return_matrix.columns.to_list()[:3])})
    table['partion_loc'] = partition_loc
    table['start_day'] = start_day
    table['end_day'] = end_day
    table['nmlz_days'] = nmlz_days
    table['trl_day'] = trl

    table2 = pd.concat(
        [table, pd.DataFrame({'年化收益率 （全时期）': annual_ret, '超额收益（全时期）：': excess_return, '夏普比率 （全时期）': sharp,
                              '最大回撤率 （全时期）': maximum_draw, '年化收益率 （前2/3时期）': annual_ret_2,
                              '夏普比率 （前2/3时期）': sharp_2, '最大回撤率 （前2/3时期）': maximum_draw_2,
                              '年化收益率 （后1/3时期）': annual_ret_3,
                              '夏普比率 （后1/3时期）': sharp_3, '最大回撤率 （后1/3时期）': maximum_draw_3,
                              'IC值': [IC_mean, IC_mean, IC_mean],
                              'ICIR': [ICIR, ICIR, ICIR]})], axis=1)
    return table2


def detail_table(total_return_matrix, top_return_matrix, bottom_return_matrix, portfolio_return_matrix, ic_df, trl,
                 nmlz_days, factor_1_name, factor_2_name, method='', sector_member=''):
    return_matrix = pd.DataFrame(
        [total_return_matrix, top_return_matrix, bottom_return_matrix, portfolio_return_matrix]).T
    return_matrix.columns = ['LT_SB', "Long_top", "Long_bottom", "Portfolio"]
    # 收益表格
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        table = table_return(return_matrix, ic_df, method, trl, nmlz_days, sector_member, factor_1_name, factor_2_name)
    return table, return_matrix


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


def mkdir(parent_path, start_date, end_date, nmlz_days, trl, method, factor_1_name, factor_2_name):
    dir = parent_path + '\\' + str(start_date.strftime("%Y-%m")) + '_' + str(
        end_date.strftime("%Y-%m")) + '_' + factor_1_name + '&&' + factor_2_name + '-' + partition_loc + '_trl' + str(trl)
    # 判断父文件夹是否存在，并创建父文件夹
    if not os.path.exists(dir):
        os.makedirs(dir)
    # 创建子文件夹
    # 判断是否有重名的文件，有：则创造新的文件夹以呈放文件
    i = 0
    son_dir = dir + '\\nmlz_days' + str(nmlz_days)
    while os.path.exists(son_dir + '\\table_' + method + '.csv'):
        i = i + 1
        if not os.path.exists(son_dir + '_' + str(i) + '\\table_' + method + '.csv'):
            son_dir = son_dir + '_' + str(i)
            break
    # 若准备放文件
    if not os.path.exists(son_dir):
        os.makedirs(son_dir)
    return son_dir  # 返回子目录的路径


def set_sector_dir(sector_memeber_name):
    parent_path = save_path + '\\' + sector_memeber_name
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    return parent_path


def get_sector_number_dummy_matrix(sector_member_name):
    if sector_member_name == '中证500':
        return timing(get_China_Securities_Index500().replace(np.nan, False).replace(1.0, True), start_date, end_date)
    elif sector_member_name == '中证1000':
        return timing(get_China_Securities_Index1000().replace(np.nan, False).replace(1.0, True), start_date, end_date)
    elif sector_member_name == '中证全指':
        return timing(get_Comprehensive_CSI().replace(np.nan, False).replace(1.0, True), start_date, end_date)
    elif sector_member_name == '国证2000':
        return timing(get_National_Certificate20000().replace(np.nan, False).replace(1.0, True), start_date, end_date)
    elif sector_member_name == '沪深300':
        return timing(get_Shanghai_Shenzhen_300_Index().replace(np.nan, False).replace(1.0, True), start_date, end_date)
    else:
        print('指数名称输入错误')
        return None


def get_factor_matrix(factor_name):
    if factor_name == 'circ_mv':
        return timing(get_circ_mv().shift(1), start_date, end_date)
    elif factor_name == 'dv_ratio':
        return timing(get_dv_ratio().shift(1), start_date, end_date)
    elif factor_name == 'dv_ttm':
        return timing(get_dv_ttm().shift(1), start_date, end_date)
    elif factor_name == 'float_share':
        return timing(get_float_share().shift(1), start_date, end_date)
    elif factor_name == 'free_share':
        return timing(get_free_share().shift(1), start_date, end_date)
    elif factor_name == 'pb':
        return timing(get_pb().shift(1), start_date, end_date)
    elif factor_name == 'pe':
        return timing(get_pe().shift(1), start_date, end_date)
    elif factor_name == 'pe_ttm':
        return timing(get_pe_ttm().shift(1), start_date, end_date)
    elif factor_name == 'ps':
        return timing(get_ps().shift(1), start_date, end_date)
    elif factor_name == 'ps_ttm':
        return timing(get_ps_ttm().shift(1), start_date, end_date)
    elif factor_name == 'total_mv':
        return timing(get_total_mv().shift(1), start_date, end_date)
    elif factor_name == 'total_share':
        return timing(get_total_share().shift(1), start_date, end_date)
    elif factor_name == 'turnover_rate':
        return timing(get_turnover_rate().shift(1), start_date, end_date)
    elif factor_name == 'turnover_rate_f':
        return timing(get_turnover_rate_f().shift(1), start_date, end_date)
    elif factor_name == 'volume_ratio':
        return timing(get_volume_ratio().shift(1), start_date, end_date)
    else:
        print('获取的方法不存在')


# 生成四个矩阵(dataframe)：收益率，是否为指定成分股的dummy，市值， 波动率
def run_back_testing_new(factor_1_name, factor_2_name):
    try:
        # T_read1 = time.perf_counter()

        plot_dict_dict = {}

        # 获取数据矩阵，并降低精度
        ret = timing(get_ret_matrix(), start_date, end_date).astype('float16')
        dummy = get_sector_number_dummy_matrix(sector_member)
        parent_path = set_sector_dir(sector_member)

        # 锁定时间区间
        _factor_1_matrix = get_factor_matrix(factor_1_name).apply(np.log1p)
        _factor_2_matrix = get_factor_matrix(factor_2_name).apply(np.log1p)

        # 用于装参数表格的list
        df_table_list = []
        # T_read2 = time.perf_counter()
        # print("读取数据用时：", T_read2 - T_read1)

        # 计算新因子，并计算其他变量
        res_list = []
        for res in compute(ret, dummy, _factor_1_matrix, _factor_2_matrix):  # 把compute的数据装载到list里，可以提前释放内存。
            # 计算持仓矩阵和最终的收益率
            portfolio, ret_total, ret_boxes_df, ret_top, ret_bot, method, _factor_2_new, dummy_new, ret_new, ret_portfolio, trl, nmlz_days = res

            # 创建一个文件夹，用于装不同方法的数据
            dir = mkdir(parent_path, start_date=start_date, end_date=end_date, nmlz_days=nmlz_days, trl=trl,
                        method=method, factor_1_name= factor_1_name, factor_2_name= factor_2_name)

            # 计算IC值
            ic_df = calculate_ic(_factor_2_new, ret_new)

            # 参数汇总表
            detail_tab, return_matrix = detail_table(total_return_matrix=ret_total, top_return_matrix=ret_top,
                                                     bottom_return_matrix=ret_bot,
                                                     portfolio_return_matrix=ret_portfolio, ic_df=ic_df, trl=trl,
                                                     nmlz_days=nmlz_days, factor_1_name=factor_1_name, factor_2_name=factor_2_name,
                                                     method=method, sector_member=sector_member)

            # streamlit需要用的python变量打包
            plot_dict = {'return_matrix': return_matrix, 'ret_boxes_df': ret_boxes_df,
                         '_factor_2_new': _factor_2_new, 'dummy_new': dummy_new, 'ret_new': ret_new,
                         'factor_name1': factor_1_name, 'factor_name2': factor_2_name}
            plot_dict_dict[method] = plot_dict

            # pickle表格
            pickle_path = dir + '\\table_' + method + str('.csv')
            detail_tab.to_csv(pickle_path, index=False, encoding='gbk')

            df_table_list.append(detail_tab)
            # 循环结束

        # pickle表格
        with open(dir + '\\' + 'test.pkl', 'wb') as f:
            pickle.dump(plot_dict_dict, f, 0)
        return df_table_list

    except Exception as e:
        traceback.print_exc()
        return None
