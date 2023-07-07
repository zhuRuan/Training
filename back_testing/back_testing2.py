# coding=utf-8
import time
import warnings

import pandas
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

pandas.set_option('display.max_rows', 200)
pandas.set_option('display.max_columns', 30)


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props = props.astype(np.uint8, errors='ignore')
                    elif mx < 65535:
                        props = props.astype(np.uint16, errors='ignore')
                    elif mx < 4294967295:
                        props = props.astype(np.uint32, errors='ignore')
                    else:
                        props = props.astype(np.uint64, errors='ignore')
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props = props.astype(np.int8, errors='ignore')
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props = props.astype(np.int16, errors='ignore')
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props = props.astype(np.int32, errors='ignore')
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props = props.astype(np.int64, errors='ignore')

                        # Make float datatypes 32 bit
            else:
                props = props.astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")
            break

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


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
                 factor_2_name, partition_loc):
    '''生成三个部分的收益分析表格'''

    annual_ret, sharp, maximum_draw, excess_return = annual_revenue_total(return_matrix=return_matrix)
    annual_ret_2, sharp_2, maximum_draw_2 = annual_revenue(
        return_matrix=return_matrix.iloc[:2 * int(len(return_matrix) / 3)])
    annual_ret_3, sharp_3, maximum_draw_3 = annual_revenue(
        return_matrix=return_matrix.iloc[2 * int(len(return_matrix) / 3):])

    # 计算IC和IC/IR
    IC_mean = ic_df.mean(axis=0).iloc[0]
    ICIR = np.round(IC_mean / (ic_df.std(axis=0).iloc[0] + 1e-8), 4)
    IC_mean = np.round(IC_mean, 4)

    # 构造dataframe表格
    table = pd.DataFrame({'因子名称': [factor_1_name, factor_1_name, factor_1_name],
                          '用作条件的因子': [factor_2_name, factor_2_name, factor_2_name],
                          '使用的参数': [method, method, method], '归属指数': [sector_member, sector_member, sector_member],
                          '科目类别': list(return_matrix.columns.to_list()[:3])})
    table['partion_loc'] = partition_loc
    table['start_day'] = start_day
    table['end_day'] = end_day
    table['trl_day'] = trl
    table['nmlz_days'] = nmlz_days

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
                 nmlz_days, factor_1_name, factor_2_name, method='', sector_member='', partition_loc=''):
    '''
    生成参数汇总表
    :param total_return_matrix:
    :param top_return_matrix:
    :param bottom_return_matrix:
    :param portfolio_return_matrix:
    :param ic_df:
    :param trl:
    :param nmlz_days:
    :param factor_1_name:
    :param factor_2_name:
    :param method:
    :param sector_member:
    :return:
    '''
    return_matrix = pd.DataFrame(
        [total_return_matrix, top_return_matrix, bottom_return_matrix, portfolio_return_matrix]).T
    return_matrix.columns = ['LT_SB', "Long_top", "Long_bottom", "Portfolio"]
    # 收益表格
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        table = table_return(return_matrix, ic_df, method, trl, nmlz_days, sector_member, factor_1_name, factor_2_name,
                             partition_loc)
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


def mk_dir(path):
    # 判断父文件夹（time_period）是否存在，并创建父文件夹
    if not os.path.exists(path):
        os.makedirs(path)


def set_factor_dir(parent_path, start_date, end_date, factor_1_name, factor_2_name):
    # 判断父文件夹（time_period）是否存在，并创建父文件夹
    dir1 = parent_path + '\\' + str(start_date.strftime("%Y-%m")) + '_' + str(end_date.strftime("%Y-%m"))
    mk_dir(dir1)

    # 创建因子名文件夹
    dir2 = dir1 + '\\' + factor_1_name + '&&' + factor_2_name
    mk_dir(dir2)
    return dir2


def mk_all_dir(dir2, nmlz_days, trl, method, partition_loc):
    # 创建TOP、BOTTOM文件夹
    dir3 = dir2 + '\\' + partition_loc
    mk_dir(dir3)

    # 创建trl文件夹
    dir4 = dir3 + "\\" + 'trl_days_' + str(trl)
    mk_dir(dir4)

    # 创建nmlz文件夹
    # 判断是否有重名的文件，有：则创造新的nmlz文件夹以呈放文件
    i = 0
    son_dir_path = dir4 + '\\nmlz_days_' + str(nmlz_days)
    while os.path.exists(son_dir_path + '\\table_' + method + '.csv'):
        i = i + 1
        if not os.path.exists(son_dir_path + '_' + str(i) + '\\table_' + method + '.csv'):
            son_dir_path = son_dir_path + '_' + str(i)
            break
    # 若准备放文件
    if not os.path.exists(son_dir_path):
        os.makedirs(son_dir_path)
    return son_dir_path  # 返回子目录的路径


def set_sector_dir(sector_memeber_name):
    parent_path = save_path + '\\' + sector_memeber_name
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    return parent_path


def get_sector_number_dummy_matrix(sector_member_name, start_date, end_date):
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


def multi_process_data_analysis(_input):
    res, factor_1_name, factor_2_name, dir2 = _input
    # 计算持仓矩阵和最终的收益率
    ret_total, ret_boxes_df, ret_top, ret_bot, method, _factor_2_new, ret_new, ret_portfolio, trl_days, nmlz_days, partition_loc = res

    # 创建一个文件夹，用于装不同方法的数据
    save_dir_path = mk_all_dir(dir2, nmlz_days=nmlz_days, trl=trl_days, method=method, partition_loc=partition_loc)

    # 计算IC值
    ic_df = calculate_ic(_factor_2_new, ret_new.shift(1))

    # 参数汇总表
    detail_tab, return_matrix = detail_table(total_return_matrix=ret_total, top_return_matrix=ret_top,
                                             bottom_return_matrix=ret_bot,
                                             portfolio_return_matrix=ret_portfolio, ic_df=ic_df, trl=trl_days,
                                             nmlz_days=nmlz_days, factor_1_name=factor_1_name,
                                             factor_2_name=factor_2_name,
                                             method=method, sector_member=sector_member, partition_loc=partition_loc)

    # streamlit需要用的python变量打包
    plot_dict = {'method': method, 'trl_days': trl_days, 'nmlz_days': nmlz_days, 'return_matrix': return_matrix,
                 'ret_boxes_df': ret_boxes_df, '_factor_2_new': _factor_2_new, 'ret_new': ret_new,
                 'factor_name1': factor_1_name, 'factor_name2': factor_2_name, }

    return detail_tab, plot_dict, save_dir_path, method, trl_days, nmlz_days, partition_loc
    # 循环结束



def run_back_testing_new(factor_1_name, factor_2_name, calc_method, nmlz_days):
    '''
    生成四个矩阵(dataframe)：执行回测，输出一个df_list，并在收益率超过0.2时存储一个pkl。
    :param factor_1_name: 因子1的名称（用于执行回测因子构建第一步：筛选符合条件的日期）
    :param factor_2_name: 因子2的名称（用于利用指定日期的数据生成的因子数据）
    :param calc_method: 使用的计算方法
    :param nmlz_days: 归一化的天数
    :return: 一个dataframe表格
    '''
    try:
        # 获取数据矩阵，并降低精度
        dummy = get_sector_number_dummy_matrix(sector_member, start_date, end_date)
        ret = timing(get_ret_matrix(), start_date, end_date)[dummy].astype('float16')
        parent_path = set_sector_dir(sector_member)

        # 锁定时间区间
        _factor_1_matrix = timing(get_factor_matrix(factor_1_name), start_date, end_date)[dummy]
        _factor_2_matrix = timing(get_factor_matrix(factor_2_name), start_date, end_date)[dummy]

        # 预处理
        _factor_1_matrix_pre = (_factor_1_matrix - _factor_1_matrix.min(axis=1).min(axis=0)).apply(np.log1p).fillna(method='backfill',axis=0, limit=10)
        _factor_2_matrix_pre = (_factor_2_matrix - _factor_2_matrix.min(axis=1).min(axis=0)).apply(np.log1p).fillna(method='backfill',axis=0, limit=10)


        # 用于装参数表格的list
        df_table_list = []

        dir2 = set_factor_dir(parent_path, start_date, end_date, factor_1_name, factor_2_name)
        # 计算新因子，把compute的数据装载到list里，可以提前释放内存。
        new_res_list = []
        for return_items in compute(ret, _factor_1_matrix_pre, _factor_2_matrix_pre, calc_method, nmlz_days):
            ret_total, ret_boxes_df, ret_top, ret_bot, method, _factor_2_new, ret_new, ret_portfolio, trl, nmlz_days, partition_loc = return_items
            new_res_list.append(((ret_total, ret_boxes_df, ret_top, ret_bot, method, _factor_2_new, ret_new,
                                  ret_portfolio, trl, nmlz_days, partition_loc), factor_1_name, factor_2_name, dir2))
        # 多进程运算，得到所需变量
        with ProcessPoolExecutor(max_workers=cpu_number) as executor:
            return_list = executor.map(multi_process_data_analysis, new_res_list)

        # 将变量打包
        for _a in return_list:
            detail_tab, plot_dict, save_dir_path, method, trl_days, nmlz_days, partition_loc = _a

            # pickle表格
            pickle_path = save_dir_path + '\\table_' + method + str('.csv')
            detail_tab.to_csv(pickle_path, index=False, encoding='gbk')
            df_table_list.append(detail_tab)

            # pickle Python变量，便于调取数据
            if (detail_tab['年化收益率 （全时期）'].str.strip('%').astype(float) / 100 > 0.2).any():
                python_variable_path = save_dir_path + '\\' + method + '.pkl'
                # 存储一个字典
                with open(python_variable_path, 'wb') as f:
                    pickle.dump(plot_dict, f, 0)
        return df_table_list

    except Exception as e:
        traceback.print_exc()
        return None
