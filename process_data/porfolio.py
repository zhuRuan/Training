import pandas as pd
import numpy as np


# 多列滚动函数
# handle对滚动的数据框进行处理
def handle(x, df_A: pd.DataFrame, name, n):
    df = df_A[name].iloc[x:x + n, :]
    rank_df_row = df.rank(axis=0, method='min')

    # return rank_df_row
    return rank_df_row


# group_rolling 进行滚动
# n：滚动的行数
# df：目标数据框
# name：要滚动的列名
def group_rolling(n, df, name):
    df_B = pd.DataFrame()
    for i in range(len(df) - n + 1):
        df_B = pd.concat((df_B, handle(x=i, df_A=df, name=name, n=n)), axis=0)
    return df_B


def select_CAP(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl):
    # 按回溯期内VOL最小的lambda天对应的CAP，求均值
    CAP_mean_matrix = pd.DataFrame(columns=CAP_matrix.columns)
    for x in range(0, len(CAP_matrix)-trl+1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * 30:x * 30 + trl, :].copy(deep=True).reset_index(drop=True)
        c = a * b
        CAP_mean_matrix = pd.concat((CAP_mean_matrix, c.mean().to_frame().T), axis=0)
    return CAP_mean_matrix


def computing_1(CAP_matrix: pd.DataFrame, dummy: pd.DataFrame, lamda, boxes):
    '''
    matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
    lamda:数字，决定做多做空多少比例的数字
    '''
    # 计算M1:对所有数字进行排序，nan未计入
    matrix = CAP_matrix[dummy]
    rank_matrix = matrix.rank(axis=1, method='min')

    # 生成M2：找到每一行最大的数字
    max_numbers = rank_matrix.max(axis=1)
    max_numbers_df = max_numbers.to_frame()
    max_numbers_matrix = pd.DataFrame(np.repeat(max_numbers_df.values, len(rank_matrix.columns), axis=1))

    # 生成M3：对排位进行boxes划分
    boxes_list = []
    for i in range(0, boxes):
        box_1 = rank_matrix >= (i / boxes * max_numbers_matrix)  # .where(condition, axis=1)
        box_2 = rank_matrix[box_1] < ((i + 1) / boxes * max_numbers_matrix)
        boxes_list.append(box_2)
    m_top = rank_matrix >= (1 - lamda) * max_numbers_matrix
    m_bot = rank_matrix <= lamda * max_numbers_matrix
    m3 = m_top + m_bot

    return m3, m_top, m_bot, boxes_list


def computing_2(CAP_matrix: pd.DataFrame, VOL_matrix: pd.DataFrame, dummy: pd.DataFrame, lamda, boxes, trl):
    '''
    matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
    lamda:数字，决定做多做空多少比例的数字
    '''

    # 计算M1:对指定期限内的VOL进行排序
    VOL_rolling_rank_matrix = group_rolling(n=trl, df=VOL_matrix, name=VOL_matrix.columns.to_list())
    VOL_rolling_rank_matrix = VOL_rolling_rank_matrix > round(lamda * trl)
    VOL_rolling_rank_matrix.reset_index(inplace=True, drop=True)

    CAP_matrix_new = select_CAP(CAP_matrix=CAP_matrix, true_false_matrix=VOL_rolling_rank_matrix, trl=trl).reset_index(drop=True) # 按回溯期内VOL最小的lambda天对应的CAP，求均值

    return computing_1(CAP_matrix_new,dummy,lamda,boxes)
