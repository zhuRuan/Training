import numpy as np
import pandas as pd
from back_testing.back_testing import run_back_testing

pd.set_option('display.max_columns', None)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    boxes = 3
    lamda = 0.2
    np.random.seed(3)
    run_back_testing(lamda, boxes)

    #

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
