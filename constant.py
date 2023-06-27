import datetime
from multiprocessing import cpu_count

# 起始时间设置
start_day = '20130101'
end_day = '20201231'
start_date = datetime.datetime.strptime(start_day, '%Y%m%d')
end_date = datetime.datetime.strptime(end_day, '%Y%m%d')

# 回溯期
trl_tuple = (10,)

# 归一化读取的前置天数(未设置循环）
nmlz_days_tuple = (10,)

# 做空或做多的比例
lambda_ratio = 0.2

# 分层回测使用的盒子数量
boxes_numbers = 10

# 第一个因子中选择TOP或者BOTTOM的比例
top_ratio = 0.2

# 计算方法设置
calc_method = ('std',)  # 总共有('std','std_ratio','mean_diff','mean')四种可以选择

# 成分股选择
sector_member = '中证500'  # 可选：'中证500','中证1000','中证全指','国证2000','沪深300'

# 因子设置（未实装）
factor_1_name_list = (
    'circ_mv', 'dv_ratio', 'dv_ttm', 'float_share', 'free_share', 'pe', 'pe_ttm', 'ps', 'ps_ttm', 'total_mv',
    'total_share', 'turnover_rate', 'turnover_rate_f', 'volume_ratio')  # 因子1：用于筛选可用的因子2天数
factor_2_name_list = factor_1_name_list  # 因子2：用于计算因子值

# 因子1截取位置
partition_loc_tuple = ('TOP','BOTTOM')  # 可选TOP或BOTTOM。选择TOP：选取因子1的值排名靠前的天数；选取BOTTOM：选取因子1的值靠后的天数。

# 参考基准
# 目前默认参考基准为指数的算术平均

# 运行参数控制
# 线程数量控制
cpu_number = 4

# 存储位置
save_path = 'pickle_data'
