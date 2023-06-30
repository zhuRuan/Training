import time

import pandas as pd
import numpy as np
from constant import trl_tuple, top_ratio, partition_loc_tuple, lambda_ratio, \
    boxes_numbers, cpu_number
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 30)

h = 0
j = 0
k = 0
l = 0


# 多列滚动函数
# handle对滚动的数据框进行处理
def handle(x, df_A, name, n):
    df = df_A[name].iloc[x:x + n, :]
    rank_df_row = df.rank(axis=0, method='dense', ascending=False, na_option='keep', pct=True)

    # return rank_df_row
    return rank_df_row


def group_rolling2_mean_top(df, factor_1, factor_2, trl):
    '''
    先对因子1滚动排列，获取符合要求【相对较大】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求均值，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    '''
    global h
    rank_df = factor_1.iloc[h - trl:h, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
    _result_df = factor_2.iloc[h - trl:h, :][_true_false_df]  # 获得符合要求的因子2，不符合的则为nan
    h = h + 1
    return _result_df.mean(axis=0)  # 求均值，并返回


def group_rolling2_mean_bottom(df, factor_1, factor_2, trl):
    '''
    先对因子1滚动排列，获取符合要求【相对较小！！！】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求均值，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    '''
    global h
    rank_df = factor_1.iloc[h - trl:h, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df >= 1 - top_ratio  # 截取市值较小的对应比例
    _result_df = factor_2.iloc[h - trl:h, :][_true_false_df]
    h = h + 1
    return _result_df.mean(axis=0)


def group_rolling2_mean_diff_top(df, factor_1, factor_2, trl):
    '''
    先对因子1滚动排列，获取符合要求【相对较大】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求均值，不符合要求的另求均值，两者作差，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    '''
    global j
    rank_df = factor_1.iloc[j - trl:j, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
    _result_df_pos = factor_2.iloc[j - trl:j, :][_true_false_df]
    _result_df_neg = factor_2.iloc[j - trl:j, :][~_true_false_df]  # truefalse矩阵求反
    j = j + 1
    return _result_df_pos.mean(axis=0) - _result_df_neg.mean(axis=0)


def group_rolling2_mean_diff_bottom(df, factor_1, factor_2, trl):
    """
    先对因子1滚动排列，获取符合要求【相对较小！！！】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求均值，不符合要求的另求均值，两者作差，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    """
    global j
    rank_df = factor_1.iloc[j - trl:j, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df >= 1 - top_ratio  # 截取市值较小的对应比例
    _result_df_pos = factor_2.iloc[j - trl:j, :][_true_false_df]
    _result_df_neg = factor_2.iloc[j - trl:j, :][~_true_false_df]
    j = j + 1
    return _result_df_pos.mean(axis=0) - _result_df_neg.mean(axis=0)


def group_rolling2_std_ratio_top(df, factor_1, factor_2, trl):
    """
    先对因子1滚动排列，获取符合要求【相对较大】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求std，不符合要求的另求std，两者相除，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    """
    global k
    rank_df = factor_1.iloc[k - trl:k, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
    _result_df_pos = factor_2.iloc[k - trl:k, :][_true_false_df]
    _result_df_neg = factor_2.iloc[k - trl:k, :][~_true_false_df]
    k = k + 1
    return (_result_df_pos.std(axis=0) + 1e-8) / (_result_df_neg.std(axis=0) + 1e-8)


def group_rolling2_std_ratio_bottom(df, factor_1, factor_2, trl):
    """
    先对因子1滚动排列，获取符合要求【相对较小！！！】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求std，不符合要求的另求std，两者相除，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    """
    global k
    rank_df = factor_1.iloc[k - trl:k, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df >= 1 - top_ratio  # 截取市值较小的对应比例
    _result_df_pos = factor_2.iloc[k - trl:k, :][_true_false_df]
    _result_df_neg = factor_2.iloc[k - trl:k, :][~_true_false_df]
    k = k + 1

    return (_result_df_pos.std(axis=0) + 1e-8) / (_result_df_neg.std(axis=0) + 1e-8)  # 求std的差


def group_rolling2_std_top(df, factor_1, factor_2, trl):
    '''
    先对因子1滚动排列，获取符合要求【相对较大】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求std，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    '''
    global l
    rank_df = factor_1.iloc[l - trl:l, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df <= top_ratio  # 截取市值较大的对应比例
    _result_df = factor_2.iloc[l - trl:l, :][_true_false_df]  # 获得符合要求的因子2，不符合的则为nan
    l = l + 1
    return _result_df.std(axis=0) + 1e-8  # 求std，并返回


def group_rolling2_std_bottom(df, factor_1, factor_2, trl):
    '''
    先对因子1滚动排列，获取符合要求【相对较小！！！】的日期，具体为一个truefalse矩阵，
    再对因子2按对应的日期求std，最终生成新的因子矩阵。
    :param df: 用于apply，对应的是factor_1的一行，不需要调用
    :param factor_1: 因子1的完整矩阵
    :param factor_2: 因子2的完整矩阵
    :param trl: 回溯的范围为多少天
    :return: 返回一个series，会被apply函数自动拼接为一个dataframe
    '''
    global l
    rank_df = factor_1.iloc[l - trl:l, :].rank(axis=0, method='dense', ascending=False, na_option='keep',
                                               pct=True)  # 得到指定区域每列的降序排序
    _true_false_df = rank_df >= 1 - top_ratio  # 截取市值较小的对应比例
    _result_df = factor_2.iloc[l - trl:l, :][_true_false_df]  # 获得符合要求的因子2，不符合的则为nan
    l = l + 1
    return _result_df.std(axis=0) + 1e-8  # 求std，并返回


def select_CAP_mean_diff(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    '''
    按回溯期内VOL最小的lambda天对应的CAP，求符合条件的mean与不符合条件的mean的差值
    :param CAP_matrix:
    :param true_false_matrix:
    :param trl:回溯日期
    :param method:
    :return:
    '''
    CAP_mean_matrix = pd.DataFrame(columns=CAP_matrix.columns)
    list = []
    for x in range(0, len(CAP_matrix) - trl + 1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True).replace(False, -1)
        c1 = a * b
        b2 = b.replace(True, 2).replace(-1, True).replace(2, -1)
        c2 = a * b2
        c1[c1 < 0] = np.nan
        c2[c2 < 0] = np.nan
        list.append((c1.mean() - c2.mean()))
    return pd.DataFrame(list), method


def calculate_new_factor(a):
    '''
    按照指定方法计算新的因子矩阵
    :param a:
    :return:
    '''
    A_matrix, B_matrix, trl, method, partition_loc = a
    if method == 'std':
        if partition_loc == 'TOP':
            _return = A_matrix.apply(group_rolling2_std_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        elif partition_loc == 'BOTTOM':
            _return = A_matrix.apply(group_rolling2_std_bottom, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        else:
            _return = A_matrix.apply(group_rolling2_std_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
            print('使用的partition_loc名称错误,请检查输入：', partition_loc)
    elif method == 'mean':
        if partition_loc == 'TOP':
            _return = A_matrix.apply(group_rolling2_mean_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        elif partition_loc == 'BOTTOM':
            _return = A_matrix.apply(group_rolling2_mean_bottom, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        else:
            _return = A_matrix.apply(group_rolling2_mean_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
            print('使用的partition_loc名称错误,请检查输入：', partition_loc)
    elif method == 'mean_diff':
        if partition_loc == 'TOP':
            _return = A_matrix.apply(group_rolling2_mean_diff_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        elif partition_loc == 'BOTTOM':
            _return = A_matrix.apply(group_rolling2_mean_diff_bottom, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        else:
            _return = A_matrix.apply(group_rolling2_mean_diff_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
            print('使用的partition_loc名称错误,请检查输入：', partition_loc)
    elif method == 'std_ratio':
        if partition_loc == 'TOP':
            _return = A_matrix.apply(group_rolling2_std_ratio_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        elif partition_loc == 'BOTTOM':
            _return = A_matrix.apply(group_rolling2_std_ratio_bottom, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        else:
            _return = A_matrix.apply(group_rolling2_std_ratio_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
            print('使用的partition_loc名称错误,请检查输入：', partition_loc)
    else:
        print('方法名称错误，按std_ratio进行计算，输入为：', method)
        if partition_loc == 'TOP':
            _return = A_matrix.apply(group_rolling2_std_ratio_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        elif partition_loc == 'BOTTOM':
            _return = A_matrix.apply(group_rolling2_std_ratio_bottom, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
        else:
            _return = A_matrix.apply(group_rolling2_std_ratio_top, factor_1=A_matrix, factor_2=B_matrix, trl=trl,
                                     axis=1)
            print('使用的partition_loc名称错误,请检查输入：', partition_loc)
    return _return.iloc[trl - 1:, ], method, trl, partition_loc


def normalization(matrix: pd.DataFrame, nmlz_day: int):
    '''
    归一化处理，不能引入未来数据
    :param matrix: 待处理的数据矩阵
    :param nmlz_day: 需要回顾的归一化天数
    :return: 归一化处理后的矩阵（会损失矩阵大小）
    '''
    std_matrix = matrix.rolling(window=nmlz_day, min_periods=round(1 / 3 * nmlz_day), axis=0).std() + 1e-8  # 获取滚动样本方差
    mean_matrix = matrix.rolling(window=nmlz_day, min_periods=round(1 / 3 * nmlz_day), axis=0).mean()  # 获取滚动样本的均值
    return (matrix.iloc[nmlz_day - 1:, :] - mean_matrix.iloc[nmlz_day - 1:, :]) / std_matrix.iloc[nmlz_day - 1:,
                                                                                  :]  # 对因子矩阵进行归一化


def multi_process_new_factor(A_matrix: pd.DataFrame, B_matrix: pd.DataFrame, calc_method: str):
    '''
    获取新的因子矩阵
    :param A_matrix: 用于确定回测期限内哪些日期合适的矩阵A
    :param B_matrix: 想要用来计算回顾天数的B矩阵
    :param dummy: dummy矩阵
    :param lamda: 数字，决定做多/做空多少比例的数字
    :param boxes: 数字，需要分成几个boxes
    :param trl: 回顾的时间
    :return:一个list，list的每个单元的结构为（新因子矩阵，所用的方法名称）
    '''

    # 多线程计算新的因子矩阵
    # 构建calculate_new_factor的输入参数
    input_list = []
    for partition_loc in partition_loc_tuple:
        for trl in trl_tuple:
                input_list.append((A_matrix, B_matrix, trl, calc_method,
                     partition_loc))  # 为calculate_new_factor方法准备输入参数,分别为A矩阵，B矩阵，回溯日期，方法名称
    # 多线程获取因子数据
    with ProcessPoolExecutor(max_workers=cpu_number) as executor:
        return executor.map(calculate_new_factor, input_list)  # 按回溯期内VOL最小的lambda天对应的CAP，求均值


def computing_portfolio_matrix(_x):
    '''
    计算投资组合持仓矩阵
    :param _x:
    :return:
    '''
    # 计算排序矩阵
    new_factor_matrix, method, nmlz_days, trl, partition_loc = _x  # 赋值
    new_factor_matrix_norm = normalization(new_factor_matrix, nmlz_day=nmlz_days)  # 归一化，Z-score标准化方法,会损失一部分数据
    rank_matrix = new_factor_matrix_norm.rank(axis=1, method='dense', ascending=True, na_option='keep',
                                                    pct=True)  # 对当天所有的在交易范围内且可以交易的股票进行排序，升序
    # print(new_factor_matrix_norm_dummy.mean(axis=1))
    # 生成M3：对排位进行boxes划分，即计算出不同boxes的持仓矩阵并存在列表中
    m_boxes_list = []
    for i in range(0, boxes_numbers):
        m_box_1 = rank_matrix >= i / boxes_numbers
        m_box_2 = rank_matrix[m_box_1] < (i + 1) / boxes_numbers
        m_boxes_list.append(m_box_2)
    m_top = rank_matrix >= 1 - lambda_ratio  # 因子头部矩阵
    m_bot = rank_matrix <= lambda_ratio  # 因子尾部矩阵
    m_t_B = m_top + m_bot  # 头部与尾部矩阵

    return m_t_B, m_top, m_bot, m_boxes_list, method, new_factor_matrix_norm, trl, nmlz_days, partition_loc


def multi_process_portfolio(input_list_for_portfolio, dummy: pd.DataFrame, nmlz_days):
    # 多线程计算最终的持仓矩阵
    # 构建computing_portfolio_matrix的输入参数
    input_list2 = []
    # 新的dummy矩阵，index要完全匹配
    for res in input_list_for_portfolio:
        _new_facotr_matrix, method, trl, partition_loc = res
        input_list2.append((_new_facotr_matrix, method, nmlz_days, trl, partition_loc))
    # 多线程获取持仓矩阵
    with ProcessPoolExecutor(max_workers=cpu_number) as executor:
        computing_portfolio_matrix_return_list = executor.map(computing_portfolio_matrix, input_list2)  # 返回了持仓矩阵到上级方法
    _return_list = []
    for item in computing_portfolio_matrix_return_list:
        m_t_B, m_top, m_bot, m_boxes_list, method, new_factor_matrix_norm, trl, nmlz_days, partition_loc = item
        _return_list.append(
            (m_t_B, m_top, m_bot, m_boxes_list, method, new_factor_matrix_norm, trl, nmlz_days, partition_loc))
    return _return_list


def get_portfolio(A_matrix, B_matrix, dummy, calc_method, nmlz_days):
    '''
    先计算新因子矩阵，计算投资组合对应的truefalse矩阵
    :param A_matrix: 因子A
    :param B_matrix: 因子B
    :param dummy: 指数对应的truefalse矩阵
    :return: m_t_B, m_top, m_bot, m_boxes_list, method, new_factor_matrix_norm, dummy
    '''
    # 先计算新因子
    # t_cal1 = time.perf_counter()
    _output_list_factor = multi_process_new_factor(A_matrix=A_matrix[dummy], B_matrix=B_matrix[dummy],
                                                   calc_method=calc_method)
    # t_cal2 = time.perf_counter()
    # print('因子计算用时(包含在生成持仓矩阵内)：', t_cal2 - t_cal1)

    # 再计算持仓矩阵
    return multi_process_portfolio(input_list_for_portfolio=_output_list_factor, dummy=dummy, nmlz_days= nmlz_days)
