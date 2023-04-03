import cvxpy as cp
import numpy as np

print(cp.installed_solvers())
# Problem data.
m = 2  # M个风格因子
t = 30  # 时间维度 代表行数
n = 100  # 个股数量 代表列数
np.random.seed(1)
A = np.random.randn(m, t, n)  # 风格因子数据
b = np.random.randn(t, n)  # 收益率数据
w = cp.Variable(shape=(t, n))  # 持仓权重矩阵
J = n  # 持仓矩阵权重的分母
c = 100  # 换仓权重/

constraints = [cp.abs(cp.diff(w, axis=0)) <= c]  # 持仓变化矩阵
constraints.extend([w >= 0, w <= 5 / J, cp.sum(w, axis=1) == 1])  # 持仓约束
for i in range(len(A) - 1):
    for j in range(A[i].shape[0]):
        constraints.extend([cp.sum(w[j] @ A[i][j].reshape(-1, 1)) == 0])  # 因子暴露为0
for j in range(A[len(A) - 1].shape[0]):
    constraints.extend([cp.sum(w[j] @ A[len(A) - 1][j].reshape(-1, 1)) == 1])  # 因子暴露为1
ret = cp.diag(w @ b.T)  # 组合每日收益率
total_mean_ret = cp.sum(ret) / ret.shape[0]  # 计算组合算术平均收益率
std = cp.sqrt(cp.sum(ret - np.ones(ret.shape[0]) * total_mean_ret) / (ret.shape[0] - 1))  # 计算组合收益率标准差
obj1 = cp.Maximize(std)  # 目标1：最大化持仓收益率方差

prob1 = cp.Problem(obj1, constraints)  # 问题1： 求最大化持仓收益率方差的解
result = prob1.solve('SCS')
print(w.value)
print(constraints[0].dual_value)