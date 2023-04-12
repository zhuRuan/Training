import numpy as np
import pandas as pd
import streamlit as st
from back_testing.back_testing import run_back_testing
from datetime import datetime as dt
import datetime

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # st.set_page_config(layout="wide", page_icon="🧊", page_title="回测结果展示")
    # st.title("回测结果展示")
    # st.markdown('当前源代码更新日期为：**:blue[2023年4月3日]**', unsafe_allow_html=False)
    # sidebar = st.sidebar
    # now_time = dt.now()
    #
    # if 'first_visit' not in st.session_state:
    #     first_visit = True
    # else:
    #     first_visit = False
    # # 初始化全局配置
    # if first_visit:
    #     st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(
    #         hours=8)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
    #     st.balloons()  # 第一次访问时才会放气
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")

    boxes = 10
    lamda = 0.2
    lag = 1
    rows = 100
    columns = 200
    np.random.seed(3)

    run_back_testing(lamda, boxes, lag, rows, columns)



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
