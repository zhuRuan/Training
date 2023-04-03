import numpy as np
import pandas as pd
import streamlit as st
from back_testing.back_testing import run_back_testing
from datetime import datetime as dt
import datetime

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    boxes = 10
    lamda = 0.2
    lag = 1
    rows = 100
    columns = 500
    np.random.seed(3)
    run_back_testing(lamda, boxes, lag, rows, columns)



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
