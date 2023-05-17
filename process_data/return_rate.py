import pandas as pd
from process_data.porfolio import *


def computing_profolio_return_rate(m3: pd.DataFrame, ret: pd.DataFrame):
    '''
    根据True-False矩阵计算组合收益率
    '''
    _true_false = m3.replace(True,1)
    invest_numbers = _true_false.sum(axis=1)
    result = ret * _true_false  # 收益率乘以m3
    mean_ret = result.sum(axis=1) / invest_numbers
    return mean_ret


def computing(ret: pd.DataFrame, dummy: pd.DataFrame, CAP: pd.DataFrame, lamda, boxes):
    '''
    计算
    输入：
    ret：收益率矩阵
    dummy：持仓矩阵
    CAP：市值矩阵
    lamda：做多和做空比率
    '''
    m3, m_top, m_bot, boxes_list = computing_1(CAP, dummy, lamda, boxes)  # 得到 True False 持仓矩阵

    # 计算不同boxes的收益率
    ret_df = pd.DataFrame()
    columns = []
    i = 0
    for box in boxes_list:
        ret_box = computing_profolio_return_rate(box, ret)
        ret_df = pd.concat([ret_df, ret_box], axis=1, ignore_index=True)
        columns.append('box' + str(i))
        i += 1
    ret_df.columns = columns

    # 计算前分位数和后分位数头寸的收益
    ret_top = computing_profolio_return_rate(m_top, ret)
    ret_bot = computing_profolio_return_rate(m_bot, ret)

    # 计算收益率
    total_ret = ret_top - ret_bot

    return m3, total_ret, ret_df, ret_top, ret_bot


def multi_thread_cp2(ret_matrix: pd.DataFrame, dummy: pd.DataFrame, CAP: pd.DataFrame, VOL: pd.DataFrame, lamda, boxes,
                     trl):
    '''
    多线程计算的中间流程
    :param ret_matrix:
    :param dummy:
    :param CAP:
    :param VOL:
    :param lamda:
    :param boxes:
    :param trl:
    :return:
    '''
    # 计算各种持仓的矩阵
    output_list = computing_2(CAP, VOL, dummy, lamda, boxes, trl)

    # 多线程
    input_list = []
    ret_matrix_cut = ret_matrix.iloc[nmlz_day-1:,:].copy(deep=True)
    # 准备多线程的输入
    for output in output_list:
        m3, m_top, m_bot, boxes_list, method, CAP_matrix, dummy_new = output
        input_list.append((ret_matrix_cut, dummy_new, CAP_matrix, VOL, lamda, boxes, trl, m3, m_top, m_bot, boxes_list, method))
    with ThreadPoolExecutor(max_workers=None) as executor:
        #
        return executor.map(computing2, input_list)
    # m3, m_top, m_bot, boxes_list = computing2(ret_matrix, dummy, CAP, VOL, lamda, boxes, trl, output_list)


def computing2(_x):
    '''
    计算
    输入：
    ret：收益率矩阵
    dummy：持仓矩阵
    CAP：市值矩阵
    lamda：做多和做空比率
    '''
    ret_matrix, dummy, CAP, VOL, lamda, boxes, trl, m3, m_top, m_bot, boxes_list, method = _x
    # m3, m_top, m_bot, boxes_list = computing_2(CAP, VOL, dummy, lamda, boxes, trl)  # 得到 True False 持仓矩阵

    # 计算不同boxes的收益率
    ret = ret_matrix.iloc[trl - 1:, :].copy(deep=True)
    columns = []
    i = 0
    ret_list = []
    for box in boxes_list:
        ret_box = computing_profolio_return_rate(box, ret)
        columns.append('box' + str(i))
        ret_list.append(ret_box)
        i += 1
    ret_df = pd.concat(ret_list,axis=1)
    ret_df.columns = columns

    # 计算前分位数和后分位数头寸的收益
    ret_top = computing_profolio_return_rate(m_top, ret)
    ret_bot = computing_profolio_return_rate(m_bot, ret)

    # 计算收益率，买TOP，卖空BOTTOM
    total_ret = ret_top - ret_bot

    return m3, total_ret, ret_df, ret_top, ret_bot, method, CAP, dummy, ret

