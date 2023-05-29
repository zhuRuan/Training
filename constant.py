#起始时间设置
start_day = '20160101'
end_day= '20160531'

#回溯期
trl_tuple = (10,)

# 归一化读取的前置天数
nmlz_day = 10

# 做空或做多的比例
ratio = 0.1

# 第一个因子中选择TOP或者BOTTOM的比例
top_ratio = 0.2

# 计算方法设置
calc_method = ('std',)
#'std_ratio','mean_diff','mean') # 总共有('std','std_ratio','mean_diff','mean')四种可以选择

# 因子设置（未实装）
factor_1 = 'dv_ttm' # 因子1：用于筛选可用的因子2天数
factor_2 = 'turnover_rate' # 因子2：用于计算因子值

# 因子1截取位置
partition_loc = 'TOP' # 可选TOP或BOTTOM。选择TOP：选取因子1的值排名靠前的天数；选取BOTTOM：选取因子1的值靠后的天数。

#参考基准






