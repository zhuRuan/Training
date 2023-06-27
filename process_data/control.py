# coding=utf-8
import time

import pandas as pd
from process_data.portfolio import *
from process_data.return_rate import *


def compute(ret_matrix: pd.DataFrame, dummy: pd.DataFrame, A_matrix: pd.DataFrame, B_matrix: pd.DataFrame):
    '''
        计算持仓矩阵和组合收益率
        :param ret_matrix:
        :param dummy:
        :param A_matrix:
        :param B_matrix:
        :return:
        '''
    return_list = []
    portfolio_output_list = []
    return_rate_output_list = []

    # 计算各种持仓的矩阵
    # t_port1 = time.perf_counter()
    output_list_from_portf = get_portfolio(A_matrix, B_matrix,
                                           dummy)  # 返回m_t_B, m_top, m_bot, m_boxes_list, method, new_factor_matrix_norm, dummy
    # t_port2 = time.perf_counter()
    # print('生成持仓矩阵用时：', t_port2 - t_port1)

    # 存储结果
    for output in output_list_from_portf:
        m_t_b, m_top, m_bot, boxes_list, method, new_factor_matrix_norm, dummy_new, trl, nmlz_days,partition_loc = output
        portfolio_output_list.append((m_t_b, m_top, m_bot, boxes_list, method, new_factor_matrix_norm, dummy_new,
                                      ret_matrix.iloc[nmlz_days + trl - 2:, :].copy(deep=True), trl, nmlz_days, partition_loc))

    # 计算组合收益率
    # t_revenue1 = time.perf_counter()
    output_list_from_retur = get_return_rate(portfolio_output_list)
    # t_revenue2 = time.perf_counter()
    # print('生成组合收益率用时：', t_revenue2 - t_revenue1)

    # 返回列表整理
    for z in output_list_from_retur:
        total_ret, ret_df, ret_top, ret_bot, ret_total_portfolio = z
        return_rate_output_list.append((total_ret, ret_df, ret_top, ret_bot, ret_total_portfolio))

    for x, y in zip(portfolio_output_list, return_rate_output_list):
        m_t_b, m_top, m_bot, boxes_list, method, new_factor_matrix_norm, dummy_new, ret_matrix_cut, trl, nmlz_days,partition_loc = x
        total_ret, ret_df, ret_top, ret_bot, ret_total_portfolio = y
        return_list.append(
            (m_t_b, total_ret, ret_df, ret_top, ret_bot, method, new_factor_matrix_norm, dummy_new, ret_matrix_cut,
             ret_total_portfolio, trl, nmlz_days, partition_loc))
    return return_list
