import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


def winsorize_plot(data, vertical_lines=[]):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    _df = ax.hist(data, 100, density=True)
    color_list = ['red', 'green', 'yellow', 'black', 'gold', 'gray']
    count = 0
    for value in vertical_lines:
        ax.bar(value, 0.1, width=0.05, color=color_list[count], alpha=1)
        count += 1
    plt.show()


def return_rate_matrix():
    # 生成return_rate矩阵
    ret = pd.DataFrame((np.random.rand(20, 30) - 0.5) * 0.42)
    return ret


def dummy_matrix():
    # 生成是否为成分股的矩阵
    dummy = pd.DataFrame([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          ])

    dummy = dummy > 0
    return dummy


def CAP_matrix():
    # 生成市值矩阵
    df1 = pd.DataFrame(np.random.randint(20, 2000, (10, 28)))
    df1[28] = df1[3]
    df1[29] = df1[3]
    df2 = pd.DataFrame(np.random.randint(20, 2000, (10, 27)))
    df2[27] = df2[3]
    df2[28] = df2[3]
    df2[29] = df2[3]
    CAP = df1.append(df2, ignore_index=True)
    return CAP


'''
matrix:pd.DataFrame, 经过与dummy相乘之后的矩阵
lamda:数字，决定做多做空多少比例的数字
'''


def computing_1(matrix: pd.DataFrame, dummy: pd.DataFrame, lamda, boxes):
    # 计算M1:对所有数字进行排序，nan未计入
    matrix = matrix[dummy]
    rank_matrix = matrix.rank(axis=1, method='first')

    # 生成M2：找到每一行最大的数字
    max_numbers = rank_matrix.max(axis=1)

    # 生成M3：对排位进行boxes划分
    boxes_list = []
    for i in range(0, boxes):
        box_1 = rank_matrix >= i / boxes * max_numbers
        box_2 = rank_matrix < ((i+1) / boxes * max_numbers)
        box = box_1 + box_2
        boxes_list.append(box)
    m_top = rank_matrix >= (1 - lamda) * max_numbers
    m_bot = rank_matrix <= lamda * max_numbers
    m3 = m_top + m_bot

    return m3, m_top, m_bot, boxes_list


def get_matrices():
    # 生成三个矩阵，分别是收益率、成分股归属、市值
    ret = return_rate_matrix()
    dummy = dummy_matrix()
    CAP = CAP_matrix()
    return ret, dummy, CAP


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


'''
计算IC值
输入：
factor:因子值矩阵
ret:收益率矩阵
'''


def calculate_ic(factor: pd.DataFrame, ret: pd.DataFrame):
    ret.index = factor.index
    factor_mean = factor.mean(axis=1)
    ret_mean = ret.mean(axis=1)
    a1 = (factor - factor_mean).fillna(value=0)
    a2 = (ret - ret_mean).fillna(value=0)
    a3 = a2.transpose()
    matrix = np.dot(a1, a3)
    numbers = factor.count(axis=1)
    cov = np.diagonal(matrix) / (numbers - 1)
    std_factor = factor.std(axis=1)
    std_ret = ret.std(axis=1)

    ic = cov / (std_factor * std_ret)

    return ic


# MAD:中位数去极值
def filter_extreme_MAD(series, n=5):
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n * new_median
    min_range = median - n * new_median
    return np.clip(series, min_range, max_range, axis=1)


'''
因子暴露展示
输入：
CAP：市值矩阵
输出：
valid数量变化折线图
factor数值的分布图
factor取极值之后的分布图
'''


def exposure(CAP: pd.DataFrame):
    # 有效数字
    valid_number = CAP.count(axis=1)
    valid_number.plot()
    plt.show()

    # 直方图
    dist = pd.DataFrame(CAP.to_numpy().flatten())
    winsorize_plot(dist)

    # 去极值后的直方图
    mad_winsorize = filter_extreme_MAD(dist, 3)
    winsorize_plot(mad_winsorize)
    return None

def mono_dist(ret_list):
    # 计算加总
    ret_cum_list=[]
    for series in ret_list:
        ret_cum_list.append(series.cumprod().tail(1))
    return ret_cum_list

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    boxes = 3

    np.random.seed(3)

    # 生成三个矩阵(dataframe)：收益率，是否为指定成分股的dummy，最新市值
    ret, dummy, CAP = get_matrices()

    # 数组运算
    portfolio, ret_total, ret_list, ret_top, ret_short = computing(ret, dummy, CAP, 0.2, 3)

    print("持仓矩阵：")
    print(portfolio)

    # 净值曲线展示
    ret_cum = ret_total.cumprod()
    ret_cum.plot()
    plt.show()

    # 因子暴露
    exposure(CAP)

    # 单调性
    ic = calculate_ic(CAP[dummy].loc[:len(CAP) - 2, :], ret[dummy].loc[1:, :])
    ic_cum = ic.cumsum()
    mono_dist(ret_list)

    #

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
