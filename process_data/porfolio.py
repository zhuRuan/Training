import pandas as pd
import numpy as np

'''
matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
lamda:数字，决定做多做空多少比例的数字
'''


def computing_1(matrix: pd.DataFrame, dummy: pd.DataFrame, lamda, boxes):
    # 计算M1:对所有数字进行排序，nan未计入
    matrix = matrix[dummy]
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