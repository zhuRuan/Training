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
    # 有效数值
    valid_number = CAP.count(axis=1).rename('valid_number_CAP')

    # 直方图
    dist = pd.DataFrame(CAP.to_numpy().flatten())
    dist.columns = ['CAP']

    # 去极值后的直方图
    mad_winsorize = filter_extreme_MAD(dist, 3)
    mad_winsorize.columns = ['CAP_after_MAD']

    return valid_number, dist, mad_winsorize
