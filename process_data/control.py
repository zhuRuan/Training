# coding=utf-8
import time

import pandas as pd
from process_data.portfolio import *
from process_data.return_rate import *


def compute(ret_matrix: pd.DataFrame, dummy: pd.DataFrame, A_matrix: pd.DataFrame, B_matrix: pd.DataFrame,
            lamda, boxes, trl):
    '''
        计算持仓矩阵和组合收益率
        :param ret_matrix:
        :param dummy:
        :param CAP:
        :param VOL:
        :param lamda:
        :param boxes:
        :param trl:
        :return:
        '''
    return_list = []
    portfolio_list = []
    return_rate_list = []

    # 计算各种持仓的矩阵
    t_port1 = time.perf_counter()
    output_list_from_portf = get_portfolio(A_matrix, B_matrix, dummy, lamda, boxes,
                                           trl)  # 返回m_t_B, m_top, m_bot, m_boxes_list, method, new_factor_matrix_norm, dummy
    t_port2 = time.perf_counter()
    print('生成持仓矩阵用时：', t_port2 - t_port1)

    # 存储结果
    for output in output_list_from_portf:
        m_t_b, m_top, m_bot, boxes_list, method, new_factor_matrix_norm, dummy_new = output
        portfolio_list.append((m_t_b, m_top, m_bot, boxes_list, method, new_factor_matrix_norm, dummy_new))

    # 计算组合收益率
    t_revenue1 = time.perf_counter()
    output_list_from_retur = get_return_rate(portfolio_list, trl, ret_matrix)
    t_revenue2 = time.perf_counter()
    print('生成组合收益率用时：', t_revenue2 - t_revenue1)

    # 返回列表整理
    for z in output_list_from_retur:
        total_ret, ret_df, ret_top, ret_bot, ret_matrix_cut = z
        return_rate_list.append((total_ret, ret_df, ret_top, ret_bot, ret_matrix_cut))

    for x, y in zip(portfolio_list, return_rate_list):
        m_t_b, m_top, m_bot, boxes_list, method, new_factor_matrix_norm, dummy_new = x
        total_ret, ret_df, ret_top, ret_bot, ret_matrix_cut = y
        return_list.append(
            (m_t_b, total_ret, ret_df, ret_top, ret_bot, method, new_factor_matrix_norm, dummy_new, ret_matrix_cut))
    return return_list
