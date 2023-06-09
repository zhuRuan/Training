# coding=utf-8
import pickle

import pandas
import pandas as pd
import numpy as np
import datetime


def get_circ_mv(path='is/basic/circ_mv.pkl'):
    '''
    获取流通市值
    :return:流通市值
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def get_dv_ratio():
    '''
    获取股息率
    :return:股息率
    '''
    with open('is/basic/dv_ratio.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_dv_ttm():
    '''
    获取循环平均的股息率
    :return:
    '''
    with open('is/basic/dv_ttm.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_float_share():
    '''
    获取流通股本
    :return:流通股本
    '''
    with open('is/basic/float_share.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_free_share():
    '''
    获取自由流通股本
    :return:自由流通股本
    '''
    with open('is/basic/free_share.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_pb():
    '''
    获取市净率（总市值/净资产）
    :return:市净率（总市值/净资产）
    '''
    with open('is/basic/pb.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_pe():
    '''
    获取市盈率（总市值/净利润）
    :return:市盈率（总市值/净利润）
    '''
    with open('is/basic/pe.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_pe_ttm():
    '''
    获取市盈率（TTM）
    :return:市盈率（TTM）
    '''
    with open('is/basic/pe_ttm.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_ps():
    '''
    获取市销率
    :return:市销率
    '''
    with open('is/basic/ps.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_ps_ttm():
    '''
    获取市销率（TTM）
    :return:市销率（TTM）
    '''
    with open('is/basic/ps_ttm.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_total_mv(path='is/basic/total_mv.pkl'):
    '''
    获取总市值 （万元）
    :return:总市值 （万元）
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def get_total_share():
    '''
    获取总股本
    :return:总股本
    '''
    with open('is/basic/total_share.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_turnover_rate():
    '''
    获取换手率（%）
    :return:换手率（%）
    '''
    with open('is/basic/turnover_rate.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_turnover_rate_f():
    '''
    获取换手率（自由流通股）
    :return:换手率（自由流通股）
    '''
    with open('is/basic/turnover_rate_f.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_volume_ratio(path='is/basic/volume_ratio.pkl'):
    '''
    获取量比
    :return:量比
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def get_close_price(path='is/dq/close.pkl'):
    '''
        获取收盘价格
        :return:收盘价格
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def get_adj_factor(path='is/dq_adj_factor/adj_factor.pkl'):
    '''
       获取调整后收盘价格
       :return:收盘价格
       '''
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def get_China_Securities_Index500():
    '''
     获取所有个股是否为中证500的True_False矩阵
     :return:中证500True_False矩阵
    '''
    with open('is/sector_member/中证500.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_China_Securities_Index1000():
    '''
     获取所有个股是否为中证500的True_False矩阵
     :return:中证500True_False矩阵
    '''
    with open('is/sector_member/中证1000.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_Comprehensive_CSI():
    '''
     获取所有个股是否为中证500的True_False矩阵
     :return:中证500True_False矩阵
    '''
    with open('is/sector_member/中证全指.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_National_Certificate20000():
    '''
     获取所有个股是否为中证500的True_False矩阵
     :return:中证500True_False矩阵
    '''
    with open('is/sector_member/国证2000.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_Shanghai_Shenzhen_300_Index():
    '''
     获取所有个股是否为中证500的True_False矩阵
     :return:中证500True_False矩阵
    '''
    with open('is/sector_member/沪深300.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def get_ret_matrix():
    '''
    获取收益率矩阵
    :return: 收益率矩阵
    '''
    close = get_close_price()
    adj_factor = get_adj_factor()
    adj_close = close * adj_factor
    ret_matrix = adj_close.pct_change(periods=1, fill_method='pad')
    # ret_matrix = ret_matrix.sub(ret_matrix.median(axis=1), axis=0)

    return ret_matrix


def get_suspend():
    '''
        获取所有个股是否停牌
        :return:个股停盘与否矩阵
    '''
    with open('is/suspend/suspend.pkl', 'rb') as f:
        data = pickle.load(f)
        return data


def change(suspend):
    suspend.index = pd.to_datetime(suspend.index)
    s_date = datetime.datetime.strptime('20130606', '%Y%m%d') - datetime.timedelta(days=1)
    e_date = datetime.datetime.strptime('20131016', '%Y%m%d')
    suspend = (suspend.query('@s_date< index <@e_date'))
    return suspend
