import pandas as pd
import numpy as np
from constant import calc_method, nmlz_day, partition_loc
from concurrent.futures import ThreadPoolExecutor


# 多列滚动函数
# handle对滚动的数据框进行处理
def handle(x, df_A: pd.DataFrame, name, n):
    df = df_A[name].iloc[x:x + n, :]
    rank_df_row = df.rank(axis=1, method='dense', ascending=True, na_option='keep', pct=True)

    # return rank_df_row
    return rank_df_row


def group_rolling(n, df, name):
    '''
    进行滚动
    :param n: 滚动的行数
    :param df: 目标数据框
    :param name: 要滚动的列名
    :return:
    '''
    df_B = pd.DataFrame()
    for i in range(len(df) - n + 1):
        df_B = pd.concat((df_B, handle(x=i, df_A=df, name=name, n=n)), axis=0)
    return df_B


def select_CAP_mean(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    # 按回溯期内VOL最小的lambda天对应的CAP，求均值
    list = []
    for x in range(0, len(CAP_matrix) - trl + 1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True).replace(False, -1)
        c = a * b
        c[c < 0] = np.nan
        # c.replace(0, np.nan, inplace=True)
        # CAP_mean_matrix = pd.concat((CAP_mean_matrix, c.mean().to_frame().T), axis=0)
        list.append(c.mean())
    return pd.DataFrame(list), method


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


def select_CAP_std_ratio(A_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    '''
    按回溯期内A最小的trl天对应的日期，求符合条件的A与不符合条件的A的标准差的比率
    :param A_matrix: 待求矩阵
    :param true_false_matrix: 复合条件或者不符合条件的True_False矩阵
    :param trl: 回溯期数
    :param method: 所使用的方法
    :return:
    '''
    CAP_mean_matrix = pd.DataFrame(columns=A_matrix.columns)
    list = []
    for x in range(0, len(A_matrix) - trl + 1):
        # 获取对应的a，b矩阵
        a = A_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True).replace(False, -1)
        # 获取c1矩阵，c1符合条件的为正数，不符合的为负数。
        c1 = a * b
        # 获取c2矩阵，c2与c1相反
        b2 = b.replace(True, 2).replace(-1, True).replace(2, -1)
        c2 = a * b2
        # 清理不符合要求的数字，以免影响std运算
        c1[c1 < 0] = np.nan
        c2[c2 < 0] = np.nan
        list.append((c1.std() / c2.std()))
    return pd.DataFrame(list), method


def select_CAP_std(CAP_matrix: pd.DataFrame, true_false_matrix: pd.DataFrame, trl, method):
    # 按回溯期内VOL最小的lambda天对应的CAP，求std
    CAP_mean_matrix = pd.DataFrame(columns=CAP_matrix.columns)
    list = []
    for x in range(0, len(CAP_matrix) - trl + 1):
        a = CAP_matrix.iloc[x:x + trl, :].copy(deep=True).reset_index(drop=True)
        b = true_false_matrix.iloc[x * trl:x * trl + trl, :].copy(deep=True).reset_index(drop=True)
        b = b.replace(False, -1)
        c = a * b
        c[c < 0] = np.nan
        # CAP_mean_matrix = pd.concat((CAP_mean_matrix, c.std().to_frame().T), axis=0)
        list.append(c.std())
    return pd.DataFrame(list), method


def select_CAP(a):
    '''
    按照指定方法计算新的因子1的矩阵
    :param a:
    :return:
    '''
    A_matrix, true_false_matrix, trl, method = a
    if method == 'std':
        _return =  select_CAP_std(A_matrix, true_false_matrix, trl, method)
    elif method == 'mean':
        _return = select_CAP_mean(A_matrix, true_false_matrix, trl, method)
    elif method == 'mean_diff':
        _return = select_CAP_mean_diff(A_matrix, true_false_matrix, trl, method)
    elif method == 'std_ratio':
        _return = select_CAP_std_ratio(A_matrix, true_false_matrix, trl, method)
    else:
        _return = select_CAP_std(A_matrix, true_false_matrix, trl, method)
        print('传入方法名错误为:',method)
        print("最终按std进行处置")
    _return[0].index = ( A_matrix.index.tolist()[trl-1:])
    return _return


def normalization(matrix: pd.DataFrame, nmlz_day: int):  # 归一化处理，不能引入未来数据
    '''
    :param matrix: 待处理的数据矩阵
    :param nmlz_day: 需要回顾的归一化天数
    :return: 归一化处理后的矩阵（会损失矩阵大小）
    '''

    # 获取滚动样本方差
    std_matrix = matrix.rolling(window=nmlz_day, min_periods=round(2 / 3 * nmlz_day), axis=0).std()
    mean_matrix = matrix.rolling(window=nmlz_day, min_periods=round(2 / 3 * nmlz_day), axis=0).mean()

    return (matrix.iloc[nmlz_day - 1:, :] - mean_matrix.iloc[nmlz_day - 1:, :]) / std_matrix.iloc[nmlz_day - 1:, :]


def computing_1(_x):
    '''
    计算持仓矩阵，包括：TOP矩阵，BOTTOM矩阵，以及各个因子box的矩阵
    matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
    lamda:数字，决定做多做空多少比例的数字
    '''
    # 计算M1:对所有数字进行排序，nan未计入
    CAP_matrix, dummy, lamda, boxes, method = _x  # 赋值
    CAP_matrix_norm = normalization(CAP_matrix, nmlz_day=nmlz_day)  # 归一化，Z-score标准化方法,会损失一部分数据
    matrix = CAP_matrix_norm[dummy]  # 按照dummy矩阵判断是否是指定的成分股

    rank_matrix = matrix.rank(axis=1, method='dense', ascending=True, na_option='keep', pct=True)

    # 生成M2：找到每一行最大的数字
    max_numbers = rank_matrix.max(axis=1)
    max_numbers_df = max_numbers.to_frame()
    max_numbers_matrix = pd.DataFrame(np.repeat(max_numbers_df.values, len(rank_matrix.columns), axis=1))

    # 生成M3：对排位进行boxes划分
    boxes_list = []
    for i in range(0, boxes):
        box_1 = rank_matrix >= i / boxes
        box_2 = rank_matrix[box_1] < (i + 1) / boxes
        boxes_list.append(box_2)
    m_top = rank_matrix >= 1 - lamda
    m_bot = rank_matrix <= lamda
    m3 = m_top + m_bot

    return m3, m_top, m_bot, boxes_list, method, CAP_matrix_norm, dummy


def computing_2(CAP_matrix: pd.DataFrame, B_matrix: pd.DataFrame, dummy: pd.DataFrame, lamda, boxes, trl):
    '''
    获取排名矩阵
    :param CAP_matrix: pd.DataFrame, 最终想要使用的A矩阵
    :param B_matrix: 想要用来计算回顾天数的B矩阵
    :param dummy: dummy矩阵
    :param lamda: 数字，决定做多做空多少比例的数字
    :param boxes: 数字，需要分成几个boxes
    :param trl: 回顾的时间
    :return:
    '''

    # 计算M1:对指定期限内的VOL进行排序，生成一个排序矩阵，行数为为 trl*(n-29)
    VOL_rolling_rank_matrix = group_rolling(n=trl, df=B_matrix, name=B_matrix.columns.to_list())

    # 获得回溯期内符合要求的日期的true-false矩阵
    if partition_loc == 'TOP':  # 若为TOP，则选取靠上的日期
        VOL_True_False_matrix = VOL_rolling_rank_matrix > lamda
    elif partition_loc == 'BOTTOM':  # 若为BOTTOM，则选取靠下的日期
        VOL_True_False_matrix = VOL_rolling_rank_matrix < lamda
    else:
        print("partition_loc设定错误，请检查constant.py，您输入的partition_loc为：", partition_loc)
        print("正确的输入为：TOP或BOTTOM，当前按默认TOP方式输入")
        VOL_True_False_matrix = VOL_rolling_rank_matrix > lamda
    VOL_True_False_matrix.copy(deep=True).reset_index(inplace=True, drop=True)

    # 构建select_CAP的输入参数
    input_list = []
    for method in calc_method:
        input_list.append((CAP_matrix, VOL_True_False_matrix, trl, method))

    # 多线程获取
    with ThreadPoolExecutor(max_workers=None) as executor:
        CAP_matrix_new_list = executor.map(select_CAP, input_list)  # 按回溯期内VOL最小的lambda天对应的CAP，求均值


    input_list2 = []
    dummy = dummy.iloc[trl+nmlz_day-2:,:]
    for res in CAP_matrix_new_list:
        CAP_matrix, method = res
        input_list2.append((CAP_matrix, dummy, lamda, boxes, method))
    with ThreadPoolExecutor(max_workers=None) as executor:
        return executor.map(computing_1, input_list2)  # 返回了computing_1矩阵
    return
