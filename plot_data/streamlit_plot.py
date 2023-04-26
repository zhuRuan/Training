import pickle

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
from numpy import linspace
from scipy.stats.kde import gaussian_kde
from datetime import datetime as dt
import datetime

st.set_page_config(layout="wide", page_icon="🧊", page_title="回测结果展示")
st.title("回测结果展示")
st.markdown('当前源代码更新日期为：**:blue[2023年4月26日]**', unsafe_allow_html=False)
sidebar = st.sidebar
now_time = dt.now()


if 'first_visit' not in st.session_state:
    first_visit = True
else:
    first_visit = False
# 初始化全局配置
if first_visit:
    st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(
        hours=8)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
    st.balloons()  # 第一次访问时才会放气
st.write("")
st.write("")
st.write("")
st.write("")


def space(num_lines=1):  # 空格
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")


space(5)

def MaxDrawdown(return_list):
    '''最大回撤率'''
    matrix = return_list.copy().reset_index(drop=True)
    i = np.argmax(
        (np.maximum.accumulate(matrix, axis=0) - matrix) / np.maximum.accumulate(matrix))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(matrix[:i])  # 开始位置
    if not matrix.empty:
        num = (matrix[j] - matrix[i]) / matrix[j]
    else:
        num = 0
    return num


def MaxDrawdown_protfolio(return_matrix: pd.DataFrame):
    maxDrawdown_dict = {}
    maxDrawdown_list = []
    for column in list(return_matrix.columns):
        MaxDrawdown_num = MaxDrawdown(return_matrix[column])
        maxDrawdown_dict[column] = MaxDrawdown_num
        maxDrawdown_list.append(MaxDrawdown_num)
    return maxDrawdown_list

def annual_revenue(return_matrix: pd.DataFrame):
    '''计算年化收益率、夏普比率、最大回撤'''
    std_list = return_matrix.std(axis=0)
    return_series = return_matrix.iloc[-1, :]
    annualized_rate_of_return = pd.Series(
        ((np.sign(return_series.values) * np.power(abs(return_series.values), 250 / len(return_matrix))) - 1).round(3))
    return_series = return_series - 1
    sharp_series = (return_series / std_list).round(3)
    maximum_drawdown_series = pd.Series(MaxDrawdown_protfolio(return_matrix)).round(3)
    return annualized_rate_of_return.values, sharp_series.values, maximum_drawdown_series.values

def table_return(return_matrix: pd.DataFrame, ic_df: pd.DataFrame, method):
    '''生成三个部分的收益分析表格'''

    annual_ret, sharp, maximum_draw = annual_revenue(return_matrix=return_matrix)
    annual_ret_2, sharp_2, maximum_draw_2 = annual_revenue(
        return_matrix=return_matrix.iloc[:2 * int(len(return_matrix) / 3), :])
    annual_ret_3, sharp_3, maximum_draw_3 = annual_revenue(
        return_matrix=return_matrix.iloc[2 * int(len(return_matrix) / 3):, :])
    IC_mean = ic_df.mean(axis=0).round(3).iloc[0]
    ICIR = np.round(IC_mean / ic_df.std(axis=0).iloc[0], 3)
    return pd.DataFrame(
        {'因子名称': ['CAP', 'CAP', 'CAP'], '参数1': [method, method, method], '参数2': ['', '', ''],
         '科目类别': list(return_matrix.columns),
         '年化收益率 （全时期）': annual_ret, '夏普比率 （全时期）': sharp, '最大回撤率 （全时期）': maximum_draw, '年化收益率 （前2/3时期）': annual_ret_2,
         '夏普比率 （前2/3时期）': sharp_2, '最大回撤率 （前2/3时期）': maximum_draw_2, '年化收益率 （后1/3时期）': annual_ret_3,
         '夏普比率 （后1/3时期）': sharp_3, '最大回撤率 （后1/3时期）': maximum_draw_3, 'IC值': [IC_mean, IC_mean, IC_mean],
         'ICIR': [ICIR, ICIR, ICIR]})

def detail_table(total_return_matrix, top_return_matrix, bottom_return_matrix, ic_df, method = ''):
    return_matrix = pd.DataFrame([total_return_matrix, top_return_matrix, bottom_return_matrix]).T
    return_matrix.columns = ['LT_SB', "Long_top", "Long_bottom"]
    # 收益表格
    table = table_return(return_matrix, ic_df, method)
    return table, return_matrix

def selectbox(calc_method):
    option = st.selectbox('选择您要查看的因子', calc_method)


def plot_table(table, fig_title: str):
    fig = go.Figure(
        data=[go.Table(
            header=dict(values=list(table.columns),
                        line_color='darkslategray',  # 线条和填充色
                        fill_color='royalblue',
                        font=dict(color='white', size=20),
                        align='center',
                        height=80),
            cells=dict(values=table.T,
                       fill_color='lavender',
                       font_size=20,
                       align='center',
                       height=40)
        )]
    )
    fig.update_layout(width=1700,
                      title=fig_title,  # 整个图的标题
                      title_font_size=25,
                      )
    st.plotly_chart(figure_or_data=fig)


def plot_return(total_return_matrix, top_return_matrix, bottom_return_matrix, ic_df, method):
    with st.container():
        st.header("组合收益分析")
        table, return_matrix = detail_table(total_return_matrix, top_return_matrix, bottom_return_matrix, ic_df, method)
        fig = go.Figure()
        fig.update_layout(width=1600,
                          title='收益曲线',
                          title_font_size=25,
                          xaxis=dict(
                              title='期数',
                              title_font_size=20,
                              tickfont_size=20  # x轴字体大小
                          ),
                          yaxis=dict(
                              title='收益率',
                              title_font_size=20,
                              tickfont_size=20
                          ),
                          )

        # 添加数据
        fig.add_trace(go.Scatter(
            x=return_matrix.index,
            y=return_matrix['LT_SB'],
            mode='lines',  # 模式
            name='LT_SB'
        ))

        fig.add_trace(go.Scatter(
            x=return_matrix.index,
            y=return_matrix['Long_top'],
            mode='lines',  # 模式
            name='Long_top'
        ))
        fig.add_trace(go.Scatter(
            x=return_matrix.index,
            y=return_matrix['Long_bottom'],
            mode='lines',  # 模式
            name='Long_bottom'
        ))
        st.plotly_chart(figure_or_data=fig)  # 折线图

        # # pickle表格
        # pickle_path = 'pickle_data\\'+  str(list(table['因子名称'])[0]) +str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")) + str('.zip')
        # table.to_pickle(pickle_path)

        # 展示表格
        plot_table(table, '收益表格')

    space(4)


def kernel(dist_matrix: pd.DataFrame, trace_name='a'):
    x_range = linspace(min(dist_matrix['CAP']), max(dist_matrix['CAP']), len(dist_matrix['CAP']))
    kde = gaussian_kde(dist_matrix['CAP'])
    df = pd.DataFrame({'x_range': x_range, 'x_kde': kde(x_range)})
    trace = go.Scatter(x=df['x_range'], y=df['x_kde'], mode='markers', name=trace_name)
    return trace


def plot_exposure(valid_number_matrix, dist_matrix, dist_mad_matrix):
    with st.container():
        st.header("因子暴露")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(data_frame=valid_number_matrix, x='index', y='valid_number_CAP')
            fig.update_layout(
                title='因子有效个数',  # 整个图的标题
                title_font_size=25,
                xaxis=dict(
                    title='日期',
                    title_font_size=20,
                    tickfont_size=20  # x轴字体大小
                ),
                yaxis=dict(
                    title='有效个数',
                    title_font_size=20,
                    tickfont_size=20
                ),
            )
            # fig.update_layout(title_font_color='blue')
            st.plotly_chart(figure_or_data=fig)
        with col2:
            trace1 = kernel(dist_matrix.iloc[:int((len(dist_matrix) * 2 / 3)), :], '前三分之二')
            trace2 = kernel(dist_matrix.iloc[int((len(dist_matrix) * 2 / 3)):, :], '后三分之一')
            fig = go.Figure(data=[trace1, trace2])

            # fig = px.histogram(dist_matrix, x="CAP")
            fig.update_layout(
                title='因子分布',  # 整个图的标题
                title_font_size=25,
                xaxis=dict(
                    title_font_size=20,
                    tickfont_size=20  # x轴字体大小
                ),
                yaxis=dict(
                    title_font_size=20,
                    tickfont_size=20
                ),
            )
            st.plotly_chart(figure_or_data=fig)
        # with col2:
        #     st.subheader('MAD处理后的因子值分布')
        #     fig = px.histogram(dist_mad_matrix, x="CAP_after_MAD")
        #     st.plotly_chart(figure_or_data=fig)
    space(4)


def plot_monotonicity(mono_dist, ic_list, ic_cum_list, lag):
    with st.container():
        st.header("单调性")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.bar(data_frame=mono_dist[0], x='boxes', y=['return_rate_minus_mean'])
            fig.update_layout(
                title='因子分层单调性_滞后一期',  # 整个图的标题
                title_font_size=25,
                xaxis=dict(
                    title='盒子标签',
                    title_font_size=20,
                    tickfont_size=20  # x轴字体大小
                ),
                yaxis=dict(
                    title='收益率（去均值后）',
                    title_font_size=20,
                    tickfont_size=20
                ),
            )
            st.plotly_chart(figure_or_data=fig)
        with col2:
            fig = px.bar(data_frame=mono_dist[1], x='boxes', y=['return_rate_minus_mean'])
            fig.update_layout(
                title='因子分层单调性_滞后五期',  # 整个图的标题
                title_font_size=25,
                xaxis=dict(
                    title='盒子标签',
                    title_font_size=20,
                    tickfont_size=20  # x轴字体大小
                ),
                yaxis=dict(
                    title='收益率（去均值后）',
                    title_font_size=20,
                    tickfont_size=20
                ),
            )
            st.plotly_chart(figure_or_data=fig)
        with col3:
            fig = px.bar(data_frame=mono_dist[2], x='boxes', y=['return_rate_minus_mean'])
            fig.update_layout(
                title='因子分层单调性_滞后二十期',  # 整个图的标题
                title_font_size=25,
                xaxis=dict(
                    title='盒子标签',
                    title_font_size=20,
                    tickfont_size=20  # x轴字体大小
                ),
                yaxis=dict(
                    title='收益率（去均值后）',
                    title_font_size=20,
                    tickfont_size=20
                ),
            )
            st.plotly_chart(figure_or_data=fig)
        trace1 = go.Bar(
            x=list(ic_list.index),
            y=ic_list['IC_CAP'],
            name='IC值'
        )
        trace2 = go.Scatter(
            x=list(ic_cum_list[0].index),
            y=ic_cum_list[0]['IC_CUM_CAP'],
            name='IC累计值_L1'
        )
        trace3 = go.Scatter(
            x=list(ic_cum_list[1].index),
            y=ic_cum_list[1]['IC_CUM_CAP'],
            name='IC累计值_L5'
        )
        trace4 = go.Scatter(
            x=list(ic_cum_list[2].index),
            y=ic_cum_list[2]['IC_CUM_CAP'],
            name='IC累计值_L20'
        )
        data = [trace1, trace2, trace3, trace4]
        layout = go.Layout({"template": 'simple_white',
                            "title": {"text": 'IC值与IC累计值'}, 'title_font_size': 25,
                            "xaxis": {"title": {"text": "期数"}, "title_font_size": 20, "tickfont_size": 20},
                            "yaxis": {"title": {"text": "IC值"}, "title_font_size": 20, "tickfont_size": 20},
                            "yaxis2": {'anchor': 'x', "overlaying": 'y', "side": 'right'},  # 设置坐标轴的格式，一般次坐标轴在右侧
                            "legend": {"title": {"text": ""}, "x": 0.9, "y": 1.1},
                            "width": 1600,
                            "height": 900 * 0.618})
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(figure_or_data=fig)
    space(4)


# 净值曲线展示
path = 'D:\Ruiwen\PythonProject\Training\pickle_data'
lists = os.listdir(path)
first_name = ''
file_name = st.selectbox('您想调取什么时间段的数据？', lists)
if file_name != '':
    with open(path + '\\' + file_name + '\\' + 'test.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)
        # 选择需要的方法
        key_list = []
        for key in data.keys():
            key_list.append(key)
        method = st.selectbox("您想要观察的回测的方法是？", key_list)
        ret_total = data[method]['ret_total']
        ret_top = data[method]['ret_top']
        ret_bot = data[method]['ret_bot']
        ic = data[method]['ic_df']
        valid_number_matrix = data[method]['valid_number_matrix']
        dist_matrix = data[method]['dist_matrix']
        dist_mad_matrix = data[method]['dist_mad_matrix']
        mono_dist_list = data[method]['mono_dist']
        ic_cum_list = data[method]['ic_cum_list']
        _lag = data[method]['lag']
        ret_matrix = data[method]['ret_matrix']
        plot_return(total_return_matrix=(ret_total + 1).cumprod(), top_return_matrix=(ret_top + 1).cumprod(),
                    bottom_return_matrix=(ret_bot + 1).cumprod(), ic_df=ic, method=method)
        # 因子暴露展示
        plot_exposure(valid_number_matrix=valid_number_matrix, dist_matrix=dist_matrix, dist_mad_matrix=dist_mad_matrix)
        # 单调性展示
        plot_monotonicity(mono_dist=mono_dist_list, ic_list=ic, ic_cum_list=ic_cum_list, lag=_lag)
