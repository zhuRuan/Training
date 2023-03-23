import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime as dt


def space(num_lines=1):  # 空格
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")


st.set_page_config(layout="wide", page_icon="🧊", page_title="回测结果展示")
st.title("回测结果展示")
st.markdown('当前源代码更新日期为：**:blue[2023年3月23日]**', unsafe_allow_html=False)
sidebar = st.sidebar
now_time = dt.now()
space(5)


def table_return(return_matrix: pd.DataFrame):
    '''生成收益分析表格'''
    std_list = return_matrix.std(axis=0)
    return_list = return_matrix.iloc[-1, :] - 1
    sharp = return_list / std_list
    return pd.DataFrame({'sharp': sharp, 'return_rate': return_list})


def plot_return(total_return_matrix, top_return_matrix, bottom_return_matrix):
    with st.container():
        st.header("组合收益分析")
        st.subheader("收益曲线")
        return_matrix = pd.DataFrame([total_return_matrix, top_return_matrix, bottom_return_matrix]).T
        return_matrix.columns = ['total_return', "long_top_return", "short_bottom_return"]
        st.line_chart(data=return_matrix)
        st.subheader("收益表格分析")
        table = table_return(return_matrix)
        st.table(data=table)

    space(4)


def plot_exposure(valid_number_matrix, dist_matrix, dist_mad_matrix):
    with st.container():
        st.header("因子暴露")
        st.subheader("因子有效个数")
        st.line_chart(data=valid_number_matrix)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("因子分布")
            fig = px.histogram(dist_matrix, x="CAP")
            st.plotly_chart(figure_or_data=fig)
        with col2:
            st.subheader('MAD处理后的因子值分布')
            fig = px.histogram(dist_mad_matrix, x="CAP_after_MAD")
            st.plotly_chart(figure_or_data=fig)
    space(4)


def plot_monotonicity(mono_dist, ic_list, ic_cum_list):
    with st.container():
        st.header("单调性")
        st.subheader("因子分层单调性")
        fig = px.bar(data_frame=mono_dist, x='boxes', y=['return_rate'])
        st.plotly_chart(figure_or_data=fig)
        st.subheader("IC曲线")
        st.line_chart(data=ic_list)
        st.subheader("IC累计曲线")
        st.line_chart(data=ic_cum_list)
    space(4)
