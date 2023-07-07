# coding=utf-8
import numpy as np
import pandas as pd
import empyrical as ep

a = pd.DataFrame([[1,1,1,1,1,2,2,2,2,2,np.nan, np.nan, 3,3,3,3,np.nan,3,3,3,3,3,3,3.3,3,3,3,93, 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4, 5], [1,1,1,1,1,2,2,2,2,2,1, 2, 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, 4, 5], [1,1,1,1,1,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, 4, 5, 6]])
rank_a = a.rank(axis=1, method='min', ascending=True, na_option='keep', pct=True)
m_top = rank_a >= 1 - 0.2  # 因子头部矩阵
m_bot = rank_a <= 0.2  # 因子尾部矩阵
print(rank_a)
print(m_top.sum(axis=1))
print(m_bot.sum(axis=1))
