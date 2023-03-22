import numpy as np

# MAD:中位数去极值
def filter_extreme_MAD(series, n=5):
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n * new_median
    min_range = median - n * new_median
    return np.clip(series, min_range, max_range, axis=1)