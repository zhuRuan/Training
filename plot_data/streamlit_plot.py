import pickle
import sys
import time

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
from numpy import linspace
from scipy.stats import gaussian_kde
from datetime import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from empyrical import max_drawdown, sharpe_ratio, aggregate_returns, annual_return, cum_returns
import datetime

path = 'D:\Ruiwen\PythonProject\Training\pickle_data'
st.set_page_config(layout="wide", page_icon="🧊", page_title="回测结果展示")
st.title("回测结果展示")
title_str = '当前源代码更新日期为：**:blue[' + str(time.ctime(os.path.getmtime(path))) + ']**'
st.markdown(title_str, unsafe_allow_html=False)
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


# MAD:中位数去极值
def filter_extreme_MAD(series, n=5):
    t_mad1 = time.perf_counter()
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n * new_median
    min_range = median - n * new_median
    t_mad2 = time.perf_counter()
    print('MAD用时：', t_mad2 - t_mad1)
    return np.clip(series, min_range, max_range, axis=1)


def exposure(CAP: pd.DataFrame):
    '''
    因子暴露展示
    输入：
    CAP：市值矩阵
    输出：
    valid数量变化分布
    factor数值的分布
    factor取极值之后的分布
    '''
    # 有效数值
    valid_number = CAP.count(axis=1).rename('valid_number_CAP').to_frame().copy(deep=True).reset_index()

    # 直方图
    t_dist1 = time.perf_counter()
    rate = 10  # 采样速率
    dist = pd.DataFrame(CAP.to_numpy().flatten())  # 恒定速率采样后，降维至一维数组
    dist.columns = ['CAP']
    dist2 = dist.copy(deep=True)
    dist2.dropna(inplace=True, axis=0, how='any')
    dist2.reset_index(drop=True, inplace=True)
    t_dist2 = time.perf_counter()

    # # 去极值后的直方图
    # mad_winsorize = filter_extreme_MAD(dist, 3)
    # mad_winsorize.columns = ['CAP_after_MAD']

    return valid_number, dist2


def calculate_ic(factor: pd.DataFrame(), ret: pd.DataFrame()):
    '''
    计算IC值
    输入：
    factor:因子值矩阵
    ret:收益率矩阵
    '''
    _factor = factor.copy(deep=True)
    _factor = _factor.reset_index(drop=True)  # 同步坐标，否则会出现问题
    _ret = ret.copy(deep=True)
    _ret = _ret.reset_index(drop=True)

    a1 = (_factor.sub(_factor.mean(axis=1), axis=0))
    a2 = (_ret.sub(_ret.mean(axis=1), axis=0))
    ic = (a1 * a2).mean(axis=1) / (_factor.std(axis=1) + 1e-8) / (_ret.std(axis=1) + 1e-8)

    # 将ic从series变为dataframe
    ic_df = pd.DataFrame(ic)
    ic_df.columns = ['IC']
    return ic_df


def mono_dist(ret_cum_df: pd.DataFrame):
    # 计算加总
    ret_cum_df = ret_cum_df.to_frame()
    ret_cum_df['boxes'] = ret_cum_df.index
    ret_cum_df.columns = ['return_rate_minus_mean', 'boxes']
    ret_cum_df['return_rate_minus_mean'] = ret_cum_df['return_rate_minus_mean'] - ret_cum_df[
        'return_rate_minus_mean'].mean()

    return ret_cum_df


def monotonicity(ret: pd.DataFrame, factor: pd.DataFrame, ret_df):
    ic_df = calculate_ic(ret, factor)
    ic_cum = ic_df.cumsum()
    ic_cum.columns = ['IC_CUM_CAP']
    _mono_dist = mono_dist(ret_df)
    return ic_df, ic_cum, _mono_dist


def comprehensive_income_analysis_total(return_matrix: pd.DataFrame):
    '''计算年化收益率、夏普比率、最大回撤'''
    # 求出年化收益
    annualized_rate_of_return_series = annual_return(return_matrix.iloc[:, :3])
    # 将收益率变为涨跌了多少而非净值的多少
    sharp_series = pd.to_numeric(pd.Series(sharpe_ratio(return_matrix.iloc[:, :3])))
    # 求最大回撤
    maximum_drawdown_series = pd.Series(max_drawdown(return_matrix.iloc[:, :3]))
    # 求超额收益
    excess_return = annualized_rate_of_return_series - annual_return(return_matrix.iloc[:, 3])
    return annualized_rate_of_return_series.apply(lambda x: format(x, '.2%')).values, sharp_series.apply(
        lambda x: format(x, '.2f')).values, maximum_drawdown_series.apply(
        lambda x: format(x, '.2%')).values, excess_return.apply(lambda x: format(x, '.2%')).values


def comprehensive_income_analysis(return_matrix: pd.DataFrame):
    '''计算年化收益率、夏普比率、最大回撤'''
    # 求出年化收益
    annualized_rate_of_return_series = annual_return(return_matrix.iloc[:, :3])
    # 将收益率变为涨跌了多少而非净值的多少
    sharp_series = pd.to_numeric(pd.Series(sharpe_ratio(return_matrix.iloc[:, :3])))
    # 求最大回撤
    maximum_drawdown_series = pd.Series(max_drawdown(return_matrix.iloc[:, :3]))
    return annualized_rate_of_return_series.apply(lambda x: format(x, '.2%')).values, sharp_series.apply(
        lambda x: format(x, '.2f')).values, maximum_drawdown_series.apply(
        lambda x: format(x, '.2%')).values


def table_return(return_matrix: pd.DataFrame, ic_df: pd.DataFrame, method, factor_name1, factor_name2):
    '''生成三个部分的收益分析表格'''

    annual_ret, sharp, maximum_draw, excess_return = comprehensive_income_analysis_total(return_matrix=return_matrix)
    annual_ret_2, sharp_2, maximum_draw_2 = comprehensive_income_analysis(
        return_matrix=return_matrix.iloc[:2 * int(len(return_matrix) / 3), :])
    annual_ret_3, sharp_3, maximum_draw_3 = comprehensive_income_analysis(
        return_matrix=return_matrix.iloc[2 * int(len(return_matrix) / 3):, :])
    IC_mean = ic_df.mean(axis=0).round(3).iloc[0]
    ICIR = np.round(IC_mean / (ic_df.std(axis=0).iloc[0] + 1e-8), 3)
    return pd.DataFrame(
        {'因子名称': [factor_name1, factor_name1, factor_name1], '条件因子': [factor_name2, factor_name2, factor_name2],
         '参数1': [method, method, method], '科目类别': list(return_matrix.columns.to_list()[:3]), '年化收益率 （全时期）': annual_ret,
         '超额收益 （全时期）': excess_return,
         '夏普比率 （全时期）': sharp, '最大回撤率 （全时期）': maximum_draw, '年化收益率 （前2/3时期）': annual_ret_2, '夏普比率 （前2/3时期）': sharp_2,
         '最大回撤率 （前2/3时期）': maximum_draw_2, '年化收益率 （后1/3时期）': annual_ret_3, '夏普比率 （后1/3时期）': sharp_3,
         '最大回撤率 （后1/3时期）': maximum_draw_3, 'IC值': [IC_mean, IC_mean, IC_mean], 'ICIR': [ICIR, ICIR, ICIR]})


def detail_table(return_matrix, ic_df, method='', factor_name1='', factor_name2=''):
    # 收益表格
    table = table_return(return_matrix, ic_df, method, factor_name1, factor_name2)

    return_matrix = cum_returns(return_matrix)
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


def plot_return(return_matrix, ic_df, method, factor_name1, factor_name2):
    with st.container():
        st.header("组合收益分析")
        table, return_matrix = detail_table(return_matrix, ic_df, method, factor_name1, factor_name2)
        # 添加数据
        trace1 = go.Scatter(
            x=return_matrix.index,
            y=return_matrix['Long_top'],
            mode='lines',  # 模式
            name='Long_top[左轴]',
        )
        trace2 = go.Scatter(
            x=return_matrix.index,
            y=return_matrix['Long_bottom'],
            mode='lines',  # 模式
            name='Long_bottom[左轴]'
        )
        trace3 = go.Scatter(
            x=return_matrix.index,
            y=return_matrix['Portfolio'],
            mode='lines',  # 模式
            name='Portfolio[左轴]',
        )
        trace4 = go.Scatter(
            x=return_matrix.index,
            y=return_matrix['LT_SB'],
            mode='lines',  # 模式
            name='LT_SB[左轴]',
            # xaxis='x',
            # yaxis='y2'
        )

        layout = go.Layout(
            yaxis2=dict(anchor='x', overlaying='y', side='right')
        )
        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout
                        )
        fig.update_layout(width=1600,
                          title='收益曲线',
                          title_font_size=25,
                          xaxis=dict(
                              title='日期',
                              title_font_size=20,
                              tickfont_size=20  # x轴字体大小
                          ),
                          yaxis=dict(
                              title='收益率',
                              title_font_size=20,
                              tickfont_size=20
                          ),
                          )
        st.plotly_chart(figure_or_data=fig)  # 折线图

        # 展示表格
        plot_table(table, '收益表格')

    space(4)


def kernel(dist_matrix: pd.DataFrame, trace_name='a'):
    _dist_matrix = dist_matrix.copy(deep=True).reset_index(drop=True)
    x_range = linspace(dist_matrix['CAP'].median() - 3 * (dist_matrix['CAP'].std() + 1e-8),
                       dist_matrix['CAP'].median() + 3 * (dist_matrix['CAP'].std() + 1e-8), len(dist_matrix['CAP']))
    kde = gaussian_kde(dist_matrix['CAP'])
    df = pd.DataFrame({'x_range': x_range, 'x_kde': kde(x_range)})
    trace = go.Scatter(x=df['x_range'], y=df['x_kde'], mode='markers', name=trace_name)
    return trace


def plot_boxes_return(ret_boxes_df: pd.DataFrame):
    equity_curve = cum_returns(ret_boxes_df)
    data_list = []
    for column in equity_curve.columns.to_list():
        trace = go.Scatter(
            x=equity_curve.index,
            y=equity_curve[column],
            mode='lines',  # 模式
            name=column,
            xaxis='x',
            yaxis='y'
        )
        data_list.append(trace)
    layout = go.Layout(
        yaxis2=dict(anchor='x', overlaying='y', side='right')
    )
    fig = go.Figure(data=data_list, layout=layout
                    )
    fig.update_layout(width=1600,
                      title='收益曲线',
                      title_font_size=25,
                      xaxis=dict(
                          title='日期',
                          title_font_size=20,
                          tickfont_size=20  # x轴字体大小
                      ),
                      yaxis=dict(
                          title='收益率',
                          title_font_size=20,
                          tickfont_size=20
                      ),
                      )
    st.plotly_chart(figure_or_data=fig)  # 折线图


def plot_exposure(valid_number_matrix, dist_matrix: pd.DataFrame):
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
            trace1 = kernel(dist_matrix.iloc[:int((len(dist_matrix) * 2 / 3)), :].sample(
                n=min(5000, (int((len(dist_matrix) * 2 / 3))) - 1)), '前三分之二')
            trace2 = kernel(dist_matrix.iloc[int((len(dist_matrix) * 2 / 3)):, :].sample(
                n=min(5000, (int((len(dist_matrix) * 1 / 3))) - 1)), '后三分之一')
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


def plot_monotonicity(mono_dist, ic_df, ic_cum_list):
    with st.container():
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
            x=list(ic_df.index),
            y=ic_df['IC'],
            name='IC值',
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


def calculate_monotonicity(_lag):
    if _lag != 1:
        factor_matrix = _factor_2_new[dummy_new].iloc[:-(_lag - 1), :]
    else:
        factor_matrix = _factor_2_new[dummy_new]
    # T21 = time.perf_counter()
    ret_matrix = (ret_new[dummy_new] + 1).rolling(_lag).apply(np.prod) - 1
    ret_boxes_matrix = (ret_boxes_df + 1).rolling(_lag).apply(np.prod) - 1
    cum_ret_boxes_matrix = annual_return(ret_boxes_matrix)
    # T22 = time.perf_counter()
    # print('矩阵计算用时：', T22 - T21)
    _ic_df, _ic_cum, _mono_dist = monotonicity(factor=factor_matrix, ret=ret_matrix.iloc[(_lag - 1):, :],
                                               ret_df=cum_ret_boxes_matrix)
    # T23 = time.perf_counter()
    # print('单调性计算用时：', T23 - T22)
    # ic_cum_list.append(_ic_cum)
    # mono_dist_list.append(_mono_dist)
    # cum_ret_boxes_matrix_list.append(cum_ret_boxes_matrix)
    return _mono_dist, _ic_cum


def multi_process_cal_mono(lag_list):
    progress_text = "单调性计算中.请等待."
    my_bar = st.progress(0, text=progress_text)
    mono_dist_list = []
    ic_cum_list = []
    res_list = []
    for lag, i in zip(lag_list, range(len(lag_list))):
        res_list.append((calculate_monotonicity(lag)))
        my_bar.progress(i, text=progress_text)
    for res in res_list:
        mono_dist, _ic_cum = res
        mono_dist_list.append(mono_dist)
        ic_cum_list.append(_ic_cum)
    return mono_dist_list, ic_cum_list


def choose_dir(path, tips):
    '''
    选择合适的dir
    :param path: 母文件夹的路径
    :param tips: 指定的提示语
    :return: 返回子文件夹的名称，及子文件夹路径
    '''
    dir_list = os.listdir(path)
    for dir in dir_list:
        if dir.endswith('.csv') or dir.endswith('.pickle'):
            dir_list.remove(dir)
    dir = st.selectbox(tips, dir_list)
    return dir, path + '\\' + dir


# 净值曲线展示
# 选择指数
index, index_dir_path = choose_dir(path=path, tips="指数成分选择：")
time_period, time_period_dir_path = choose_dir(path=index_dir_path, tips='时间段选择：')
factor, factor_dir_path = choose_dir(path=time_period_dir_path, tips='测试因子选择：')
partition_loc, partition_loc_dir_path = choose_dir(path=factor_dir_path, tips='因子高值低值选择：')
trl_days, trl_days_dir_path = choose_dir(path=partition_loc_dir_path, tips='回溯天数选择：')
nmlz_days, nmlz_days_dir_path = choose_dir(path=trl_days_dir_path, tips='归一化天数选择')
key_list = []
with open(factor_dir_path + '\\' + 'python_variable.pkl', 'rb') as f:
    data = pickle.load(f)
    # 选择需要的方法
    for key in data[partition_loc + str(trl_days) + str(nmlz_days)].keys():
        key_list.append(key)
method = st.selectbox("您想要观察的因子2【即条件因子】回测的方法是？", key_list)
return_matrix = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['return_matrix']
ret_boxes_df = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['ret_boxes_df']
_factor_2_new = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['_factor_2_new']
dummy_new = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['dummy_new']
ret_new = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['ret_new']
factor_name1 = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['factor_name1']
factor_name2 = data[partition_loc + str(trl_days) + str(nmlz_days)][method]['factor_name2']
ic_df = calculate_ic(_factor_2_new, ret_new)
plot_return(return_matrix=return_matrix, ic_df=ic_df,
            method=method, factor_name1=factor_name1, factor_name2=factor_name2)

# 单调性
lag_list = [1, 5, 20]
ic = 0
ic_cum_list = []
mono_dist_list = []
cum_ret_boxes_matrix_list = []

# 去除dist的空值
# 计算因子暴露
with st.spinner('请等待...'):
    valid_number_matrix, dist_matrix = exposure(_factor_2_new)

# 因子暴露展示
plot_exposure(valid_number_matrix=valid_number_matrix, dist_matrix=dist_matrix)

# 单调性展示
# 按照滞后期数的循环
T3 = time.perf_counter()
st.header('单调性')
plot_boxes_return(ret_boxes_df)
# mono_dist_list, ic_cum_list = multi_process_cal_mono(lag_list)
# my_bar = st.empty()
# T4 = time.perf_counter()
# print('单调性运算用时：', T4 - T3)
# plot_monotonicity(mono_dist=mono_dist_list, ic_df=ic_df, ic_cum_list=ic_cum_list)
