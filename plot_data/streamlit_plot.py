import streamlit as st
import plotly.express as px
from datetime import datetime as dt


st.set_page_config(layout="centered", page_icon="random", page_title="回测结果展示")
st.title("回测结果展示"  )
sidebar = st.sidebar
now_time = dt.now()

def space(num_lines=1): #空格
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

def plot_exposure(valid_number_matrix, dist_matrix, dist_mad_matrix):
    with st.container():
        st.header("因子暴露")
        st.subheader("因子有效个数")
        st.line_chart(data = valid_number_matrix)
        st.subheader("因子分布")
        fig = px.histogram(dist_matrix, x="CAP")
        st.plotly_chart(figure_or_data=fig)
        st.subheader('MAD处理后的因子值分布')
        fig = px.histogram(dist_mad_matrix, x="CAP_after_MAD")
        st.plotly_chart(figure_or_data=fig)
    space(1)

def plot_monotonicity(mono_dist, ic_list, ic_cum_list):
    with st.container():
        st.header("单调性")
        st.subheader("因子分层单调性")
        st.bar_chart(data=mono_dist)
        st.subheader("IC曲线")
        st.line_chart(data=ic_list)
        st.subheader("IC累计曲线")
        st.line_chart(data=ic_cum_list)