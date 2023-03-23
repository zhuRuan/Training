import pandas as pd
import numpy as np

np.random.seed(3)


def return_rate_matrix(rows=50, columns=30):
    # 生成return_rate矩阵
    ret = pd.DataFrame(np.random.normal(size=(rows, columns), loc=0.00015, scale=0.03))
    return ret


def dummy_matrix(rows=50, columns=30):
    # 生成是否为成分股的矩阵
    list1 = []
    list_temp1 = []
    list_temp2 = []
    for j in range(columns):
        if j < columns / 2:
            list_temp1.append(1)
        else:
            list_temp1.append(0)
    for j in range(columns):
        if j > 2 and j < (int(columns / 2) + 3):
            list_temp2.append(1)
        else:
            list_temp2.append(0)
    for i in range(int(rows / 2)):
        list1.append(list_temp1)
    for j in range(int(rows / 2)):
        list1.append(list_temp2)
    dummy = pd.DataFrame(list1)
    dummy = dummy > 0
    return dummy


def CAP_matrix(rows=50, columns=30):
    # 生成市值矩阵
    df1 = pd.DataFrame(np.random.normal(1000, 300, (round(rows / 2), columns - 2)))
    df1[len(df1.columns)] = df1[3]
    df1[len(df1.columns)] = df1[3]
    df2 = pd.DataFrame(np.random.normal(1000, 300, (round(rows / 2), columns - 3)))
    df2[len(df2.columns)] = df2[3]
    df2[len(df2.columns)] = df2[3]
    df2[len(df2.columns)] = df2[3]
    CAP = pd.concat((df1, df2), axis=0, ignore_index=True)
    CAP.reindex()
    CAP[CAP < 30] = 30
    return CAP

# print(dummy_matrix())
