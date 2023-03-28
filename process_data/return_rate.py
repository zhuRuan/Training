import pandas as pd
from process_data.porfolio import *




def computing_profolio_return_rate(m3: pd.DataFrame, ret: pd.DataFrame):
    '''
    根据True-False矩阵计算组合收益率
    '''
    invest_numbers = m3.sum(axis=1)
    result = ret * m3  # 收益率乘以m3
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
        ret_df = pd.concat([ret_df, ret_box],axis=1, ignore_index=True)
        columns.append('box'+str(i))
        i += 1
    ret_df.columns = columns

    # 计算前分位数和后分位数头寸的收益
    ret_top = computing_profolio_return_rate(m_top, ret)
    ret_bot = computing_profolio_return_rate(m_bot, ret)

    # 计算收益率
    total_ret = ret_top - ret_bot

    return m3, total_ret, ret_df, ret_top, ret_bot