# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


def return_rate_matrix():
    # 生成return矩阵
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


def computing_1(matrix: pd.DataFrame, dummy: pd.DataFrame, lamda):
    # 计算M1:对所有数字进行排序，nan未计入
    matrix = matrix[dummy]
    rank_matrix = matrix.rank(axis=1)

    # 生成M2：找到每一行最大的数字
    max_numbers = rank_matrix.max(axis=1)

    # 生成M3：对排位进行划分
    m_top = rank_matrix >= (1 - lamda) * max_numbers
    m_bot = rank_matrix <= lamda * max_numbers
    m3 = m_top + m_bot

    return m3


def get_matrices():
    # 生成三个矩阵，分别是收益率、成分股归属、市值
    ret = return_rate_matrix()
    dummy = dummy_matrix()
    CAP = CAP_matrix()
    return ret, dummy, CAP


'''
计算
输入：
ret：收益率矩阵
dummy：持仓矩阵
CAP：市值矩阵
lamda：做多和做空比率
'''


def computing(ret: pd.DataFrame, dummy: pd.DataFrame, CAP: pd.DataFrame, lamda):
    m3 = computing_1(CAP, dummy, lamda)  # 得到 True False 持仓矩阵
    invest_numbers = m3.sum(axis=1)
    result = (1 + ret) * m3  # 收益率乘以m3
    mean_ret = result.sum(axis=1) / invest_numbers
    total_ret = mean_ret.cumprod()
    return m3, total_ret


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
    a1 = (factor.fillna(value=0) - factor_mean).fillna(value=0)
    a2 = (ret.fillna(value=0) - ret_mean).fillna(value=0)
    matrix = a1.dot(a2.transpose())
    cov = np.diagonal(matrix)
    std_factor = factor.std(axis=1)
    std_ret = ret.std(axis=1)

    ic = cov / std_factor / std_ret

    return ic


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    np.random.seed(3)

    # 生成三个矩阵(dataframe)：收益率，是否为指定成分股的dummy，最新市值
    ret, dummy, CAP = get_matrices()

    # 数组运算
    portfolio, ret_total = computing(ret, dummy, CAP, 0.2)

    print("持仓矩阵：")
    print(portfolio)

    # 净值曲线展示
    ret_total.plot()
    plt.show()

    # 因子暴露
    ic = calculate_ic(CAP[dummy].loc[:len(CAP) - 2, :], ret[dummy].loc[1:, :])
    print(ic)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
