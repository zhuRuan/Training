import pandas as pd
import numpy as np
from constant import calc_method
from concurrent.futures import ThreadPoolExecutor


# 多列滚动函数
# handle对滚动的数据框进行处理
def handle(x, df_A: pd.DataFrame, name, n):
    df = df_A[name].iloc[x:x + n, :]
    rank_df_row = df.rank(axis=1, method='dense', ascending=True, na_option='keep', pct=True)

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


def select_CAP_mean(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    # 按回溯期内VOL最小的lambda天对应的CAP，求均值
    CAP_mean_matrix = pd.DataFrame(columns=CAP_matrix.columns)
    for x in range(0, len(CAP_matrix) - trl + 1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True)
        c = a * b
        c.replace(0, np.nan, inplace=True)
        CAP_mean_matrix = pd.concat((CAP_mean_matrix, c.mean().to_frame().T), axis=0)
    return CAP_mean_matrix.reset_index(drop=True), method


def select_CAP_mean_diff(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    # 按回溯期内VOL最小的lambda天对应的CAP，求符合条件的mean与不符合条件的mean的差值
    CAP_mean_matrix = pd.DataFrame(columns=CAP_matrix.columns)
    for x in range(0, len(CAP_matrix) - trl + 1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True)
        c1 = a * b
        b2 = b.replace(True, 2).replace(False, 1).replace(2, False)
        c2 = a * b2
        c1.replace(0, np.nan, inplace=True)
        CAP_mean_matrix = pd.concat((CAP_mean_matrix, (c1.mean() - c2.mean()).to_frame().T), axis=0)
    return CAP_mean_matrix.reset_index(drop=True), method


def select_CAP_std_ratio(A_matrix: pd.DataFrame, B_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    # 按回溯期内B最小的lambda天对应的日期，求符合条件的A与符合条件的B的标准差的比率
    CAP_mean_matrix = pd.DataFrame(columns=A_matrix.columns)
    for x in range(0, len(A_matrix) - trl + 1):
        a = A_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = B_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        c = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True)
        a1 = a * c
        b1 = b * c
        a1.replace(0, np.nan, inplace=True)
        b1.replace(0, np.nan, inplace=True)
        CAP_mean_matrix = pd.concat((CAP_mean_matrix, (a1.std() / b1.std()).to_frame().T), axis=0)

    return CAP_mean_matrix.reset_index(drop=True), method


def select_CAP_std(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    # 按回溯期内VOL最小的lambda天对应的CAP，求std
    CAP_mean_matrix = pd.DataFrame(columns=CAP_matrix.columns)
    for x in range(0, len(CAP_matrix) - trl + 1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True)
        c = a * b
        c.replace(0, np.nan, inplace=True)
        CAP_mean_matrix = pd.concat((CAP_mean_matrix, c.std().to_frame().T), axis=0)
    return CAP_mean_matrix.reset_index(drop=True), method


def select_CAP(a):
    A_matrix, B_matrix, true_false_matrix, trl, method = a
    if method == 'std':
        return select_CAP_std(A_matrix, true_false_matrix, trl, method)
    elif method == 'mean':
        return select_CAP_mean(A_matrix, true_false_matrix, trl, method)
    elif method == 'mean_diff':
        return select_CAP_mean_diff(A_matrix, true_false_matrix, trl, method)
    elif method == 'std_ratio':
        return select_CAP_std_ratio(A_matrix, B_matrix, true_false_matrix, trl, method)


def computing_1(_x):
    '''
    matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
    lamda:数字，决定做多做空多少比例的数字
    '''
    # 计算M1:对所有数字进行排序，nan未计入

    CAP_matrix, dummy, lamda, boxes, method = _x # 赋值
    CAP_matrix_norm = (CAP_matrix - CAP_matrix.mean()) / CAP_matrix.std() # 归一化，Z-score标准化方法
    matrix = CAP_matrix_norm[dummy] # 按照dummy矩阵判断是否是指定的成分股
    rank_matrix = matrix.rank(axis=1, method='dense', ascending=True, na_option='keep', pct=True)

    # 生成M2：找到每一行最大的数字
    max_numbers = rank_matrix.max(axis=1)
    max_numbers_df = max_numbers.to_frame()
    max_numbers_matrix = pd.DataFrame(np.repeat(max_numbers_df.values, len(rank_matrix.columns), axis=1))

    # 生成M3：对排位进行boxes划分
    boxes_list = []
    for i in range(0, boxes):
        box_1 = rank_matrix >= i / boxes  # .where(condition, axis=1)
        box_2 = rank_matrix[box_1] < (i + 1) / boxes
        boxes_list.append(box_2)
    m_top = rank_matrix >= 1 - lamda
    m_bot = rank_matrix <= lamda
    m3 = m_top + m_bot

    return m3, m_top, m_bot, boxes_list, method


def computing_2(CAP_matrix: pd.DataFrame, VOL_matrix: pd.DataFrame, dummy: pd.DataFrame, lamda, boxes, trl):
    '''
    matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
    lamda:数字，决定做多做空多少比例的数字
    '''

    # 计算M1:对指定期限内的VOL进行排序，生成一个排序矩阵，行数为为 trl*(n-29)
    VOL_rolling_rank_matrix = group_rolling(n=trl, df=VOL_matrix, name=VOL_matrix.columns.to_list())
    VOL_rolling_rank_matrix = VOL_rolling_rank_matrix > lamda
    VOL_rolling_rank_matrix.reset_index(inplace=True, drop=True)

    #构建select_CAP的输入参数
    input_list = []
    for method in calc_method:
        input_list.append((CAP_matrix, VOL_matrix, VOL_rolling_rank_matrix, trl, method))

    with ThreadPoolExecutor(max_workers=None) as executor:
        CAP_matrix_new_list = executor.map(select_CAP, input_list)  # 按回溯期内VOL最小的lambda天对应的CAP，求均值

    input_list2 = []
    for res in CAP_matrix_new_list:
        CAP_matrix, method = res
        input_list2.append((CAP_matrix, dummy, lamda, boxes, method))
    with ThreadPoolExecutor(max_workers=None) as executor:
        return executor.map(computing_1, input_list2) # 返回了computing矩阵
    return
