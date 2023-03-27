import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime as dt

st.set_page_config(layout="wide", page_icon="🧊", page_title="回测结果展示")
st.title("回测结果展示")
st.markdown('当前源代码更新日期为：**:blue[2023年3月23日]**', unsafe_allow_html=False)
sidebar = st.sidebar
now_time = dt.now()


def space(num_lines=1):  # 空格
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")


space(5)


@st.cache_data
def table_return(return_matrix: pd.DataFrame):
    '''生成收益分析表格'''
    std_list = return_matrix.std(axis=0)
    return_list = return_matrix.iloc[-1, :] - 1
    sharp = return_list / std_list
    return pd.DataFrame({'Name': list(return_matrix.columns), 'sharp': sharp, 'return_rate': return_list})


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
                              title='收益率',
                              title_font_size=20,
                              tickfont_size=20  # x轴字体大小
                          ),
                          yaxis=dict(
                              title='期数',
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
        table = table_return(return_matrix)
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
                          title='收益表格分析',  # 整个图的标题
                          title_font_size=25,
                          )
        st.plotly_chart(figure_or_data=fig)
    space(4)


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
            fig = px.histogram(dist_matrix, x="CAP")
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
                            "title": {"text": 'IC值与IC累计值'},'title_font_size':25,
                            "xaxis": {"title": {"text": "期数"}, "title_font_size": 20, "tickfont_size": 20},
                            "yaxis": {"title": {"text": "IC值"}, "title_font_size": 20, "tickfont_size": 20},
                            "yaxis2": {'anchor': 'x', "overlaying": 'y', "side": 'right'},  # 设置坐标轴的格式，一般次坐标轴在右侧
                            "legend": {"title": {"text": ""}, "x": 0.9, "y": 1.1},
                            "width": 1600,
                            "height": 900 * 0.618})
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(figure_or_data=fig)
    space(4)
