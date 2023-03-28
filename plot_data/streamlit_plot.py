import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from numpy import linspace
from scipy.stats.kde import gaussian_kde
from datetime import datetime as dt

st.set_page_config(layout="wide", page_icon="🧊", page_title="回测结果展示")
st.title("回测结果展示")
st.markdown('当前源代码更新日期为：**:blue[2023年3月27日]**', unsafe_allow_html=False)
sidebar = st.sidebar
now_time = dt.now()


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
    num = (matrix[j] - matrix[i]) / matrix[j]
    return num


def MaxDrawdown_protfolio(return_matrix: pd.DataFrame):
    MaxDrawdown_dict = {}
    MaxDrawdown_list = []
    for column in list(return_matrix.columns):
        MaxDrawdown_num = MaxDrawdown(return_matrix[column])
        MaxDrawdown_dict[column] = MaxDrawdown_num
        MaxDrawdown_list.append(MaxDrawdown_num)
    return MaxDrawdown_list


@st.cache_data
def annual_revenue(return_matrix: pd.DataFrame):
    '''计算年化收益率、夏普比率、最大回撤'''
    std_list = return_matrix.std(axis=0)
    return_series = return_matrix.iloc[-1, :]
    annualized_rate_of_return = pd.Series(
        ((np.sign(return_series.values) * np.power(abs(return_series.values), 250 / len(return_matrix))) - 1).round(2))
    return_series = return_series - 1
    sharp_series = (return_series / std_list).round(2)
    maximum_drawdown_series = pd.Series(MaxDrawdown_protfolio(return_matrix)).round(2)
    return annualized_rate_of_return.values, sharp_series.values, maximum_drawdown_series.values


@st.cache_data
def table_return(return_matrix: pd.DataFrame):
    '''生成三个部分的收益分析表格'''

    annual_ret, sharp, maximum_draw = annual_revenue(return_matrix=return_matrix)
    annual_ret_2, sharp_2, maximum_draw_2 = annual_revenue(
        return_matrix=return_matrix.iloc[:2 * int(len(return_matrix) / 3), :])
    annual_ret_3, sharp_3, maximum_draw_3 = annual_revenue(
        return_matrix=return_matrix.iloc[2 * int(len(return_matrix) / 3):, :])

    return pd.DataFrame(
        {'科目类别': list(return_matrix.columns), '夏普比率': sharp, '年化收益率': annual_ret, '最大回撤率': maximum_draw}), \
           pd.DataFrame(
               {'科目类别': list(return_matrix.columns), '夏普比率': sharp_2, '年化收益率': annual_ret_2, '最大回撤率': maximum_draw_2}), \
           pd.DataFrame(
               {'科目类别': list(return_matrix.columns), '夏普比率': sharp_3, '年化收益率': annual_ret_3, '最大回撤率': maximum_draw_3})


def plot_table(table, fig_title: str):
    fig = go.Figure(
        data=[go.Table(
            header=dict(values=list(table.columns),
                        line_color='darkslategray',  # 线条和填充色
                        fill_color='royalblue',
                        font=dict(color='white', size=20),
                        align='center',
                        height=50),
            cells=dict(values=table.T,
                       fill_color='lavender',
                       font_size=20,
                       align='center',
                       height=40)
        )]
    )
    fig.update_layout(width=1600,
                      title=fig_title,  # 整个图的标题
                      title_font_size=25,
                      )
    st.plotly_chart(figure_or_data=fig)


@st.cache_data
def plot_return(total_return_matrix, top_return_matrix, bottom_return_matrix):
    with st.container():
        st.header("组合收益分析")
        return_matrix = pd.DataFrame([total_return_matrix, top_return_matrix, bottom_return_matrix]).T
        return_matrix.columns = ['LT_SB', "Long_top_return", "Long_bottom_return"]
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
            y=return_matrix['Long_top_return'],
            mode='lines',  # 模式
            name='long_top_return'
        ))
        fig.add_trace(go.Scatter(
            x=return_matrix.index,
            y=return_matrix['Long_bottom_return'],
            mode='lines',  # 模式
            name='long_bottom_return'
        ))
        st.plotly_chart(figure_or_data=fig)  # 折线图

        # 收益表格
        table, table2, table3 = table_return(return_matrix)
        plot_table(table, '全时期收益表格')
        plot_table(table2, '前三分之二时期收益表格')
        plot_table(table3, '后三分之一时期收益表格')

    space(4)


@st.cache_data
def kernel(dist_matrix: pd.DataFrame, trace_name = 'a'):
    x_range = linspace(min(dist_matrix['CAP']), max(dist_matrix['CAP']), len(dist_matrix['CAP']))
    kde = gaussian_kde(dist_matrix['CAP'])
    df = pd.DataFrame({'x_range': x_range, 'x_kde': kde(x_range)})
    trace = go.Scatter(x=df['x_range'], y=df['x_kde'], mode='markers', name=trace_name)
    return trace


@st.cache_data
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
            trace1= kernel(dist_matrix.iloc[:int((len(dist_matrix)*2/3)),:], '前三分之二')
            trace2= kernel(dist_matrix.iloc[int((len(dist_matrix)*2/3)):,:], '后三分之一')
            fig = go.Figure(data=[trace1,trace2])

            # fig = px.histogram(dist_matrix, x="CAP")
            fig.update_layout(
                title='因子分布',  # 整个图的标题
                title_font_size=25,
                xaxis=dict(
                    title='数量',
                    title_font_size=20,
                    tickfont_size=20  # x轴字体大小
                ),
                yaxis=dict(
                    title='因子值',
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


@st.cache_data
def plot_monotonicity(mono_dist, ic_list, ic_cum_list):
    with st.container():
        st.header("单调性")
        fig = px.bar(data_frame=mono_dist, x='boxes', y=['return_rate'])
        fig.update_layout(
            title='因子分层单调性',  # 整个图的标题
            title_font_size=25,
            xaxis=dict(
                title='盒子标签',
                title_font_size=20,
                tickfont_size=20  # x轴字体大小
            ),
            yaxis=dict(
                title='收益率',
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
            x=list(ic_cum_list.index),
            y=ic_cum_list['IC_CUM_CAP'],
            name='IC累计值'
        )
        data = [trace1, trace2]
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