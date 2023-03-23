import numpy as np

from process_data.monotonicity import calculate_ic
from read_data.generate_random_data import *
import time

ret = return_rate_matrix(6000, 200)
factor = CAP_matrix(6000, 200)

T1 = time.time()
list1 = calculate_ic(factor, ret)
T2 = time.time()
print('1运行时间：:%s毫秒' % ((T2 - T1) * 1000))


T3 = time.time()
ret = ret.to_numpy()
factor = factor.to_numpy()
list2 = []
for row_r, row_f in zip(ret, factor):
    r_mean = np.mean(row_r)
    f_mean = np.mean(row_f)
    a1 = (row_r - r_mean)
    a2 = (row_f - f_mean)
    cov = a1.dot(a2) / (len(row_r) -1)
    ic = cov /  (np.std(row_f) * np.std(row_r))
    list2.append(ic)
T4 = time.time()
print('2运行时间：:%s毫秒' % ((T4 - T3) * 1000))

print(list1)
print(list2)
