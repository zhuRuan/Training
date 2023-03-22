import pandas as pd
from process_data.MAD import filter_extreme_MAD

'''
因子暴露展示
输入：
CAP：市值矩阵
输出：
valid数量变化分布
factor数值的分布
factor取极值之后的分布
'''


def exposure(CAP: pd.DataFrame):
    # 有效数字
    valid_number = CAP.count(axis=1)
    valid_number.plot()
    # plt.show()

    # 直方图
    dist = pd.DataFrame(CAP.to_numpy().flatten())
    # winsorize_plot(dist)

    # 去极值后的直方图
    mad_winsorize = filter_extreme_MAD(dist, 3)
    # winsorize_plot(mad_winsorize)
    return valid_number, dist, mad_winsorize
