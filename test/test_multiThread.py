# coding=gbk
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def dataframe_ret (g):
    df = pd.DataFrame()
    df['new'] = [1,2,3]
    return df

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    list = ['a','b','c','d','e','f','g']
    list2 = []
    with ThreadPoolExecutor(max_workers=None) as executor:
        df_list = executor.map(dataframe_ret, list)
    for df in df_list:
        list2.append(df)

    df1 = pd.concat(list2, ignore_index = True)
    print(df1)