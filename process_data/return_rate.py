import pandas as pd
from process_data.porfolio import *


'''
根据True-False矩阵计算组合收益率
'''


def computing_profolio_return_rate(m3: pd.DataFrame, ret: pd.DataFrame):
    invest_numbers = m3.sum(axis=1)
    result = (1 + ret) * m3  # 收益率乘以m3
    mean_ret = result.sum(axis=1) / invest_numbers

    return mean_ret

'''
计算
输入：
ret：收益率矩阵
dummy：持仓矩阵
CAP：市值矩阵
lamda：做多和做空比率
'''


def computing(ret: pd.DataFrame, dummy: pd.DataFrame, CAP: pd.DataFrame, lamda, boxes):
    m3, m_top, m_bot, boxes_list = computing_1(CAP, dummy, lamda, boxes)  # 得到 True False 持仓矩阵

    # 计算收益率
    total_ret = computing_profolio_return_rate(m3, ret)

    # 计算不同boxes的收益率
    ret_list = []
    for box in boxes_list:
        ret_box = computing_profolio_return_rate(box, ret)
        ret_list.append(ret_box)

    # 计算前分位数和后分位数头寸的收益
    ret_top = computing_profolio_return_rate(m_top, ret)
    ret_bot = computing_profolio_return_rate(m_bot, ret)

    return m3, total_ret, ret_list, ret_top, ret_bot