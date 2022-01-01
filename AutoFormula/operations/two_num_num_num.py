# Copyright (c) 2021 Dai HBG


"""
该代码定义2_num_num_num型运算符
"""


import numpy as np
import numba as nb


# @nb.jit 存在操作list的的代码，应该无法使用numba
def tssubset(a, b, num_0, num_1, num_2):  # 时序选择算子，回溯num_0天，根据b的排序选出从num_1到num_2的子集，返回均值
    s = np.zeros(a.shape)
    for i in range(num - 1, a.shape[0]):
        for j in range(a.shape[1]):
            lst = [(a[i - k, j], b[i - k, j]) for k in range(num_0)]
            lst = sorted(lst, key=lambda x: x[1])
            lst = [lst[k][0] for k in range(num_1, num_2 + 1)]
            s[i, j] = np.mean(lst)
    s[:num_0 - 1] = np.nan
    return s


def biintraquantile(a, b, start, end, num):  # 根据b排序的a的日内分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入，返回2维矩阵
    tmp_a = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp_b = b[:, start:end + 1, :].transpose(1, 0, 2)
    arg_b = np.argsort(tmp_b, axis=0)
    if num == 1:
        pos = end - start
    elif num == 0:
        pos = 0
    else:
        pos = int(num * (end + 1 - start))
    se = arg_b[pos]  # 对应的a的quantile位
    s = np.zeros(tmp_a[0].shape)
    for i in range(end+1-start):
        s[se == i] = tmp_a[i][se == i]
    return s


def biintraquantileupmean(a, b, start, end, num):  # 根据b排序的a日内分位数上行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    tmp_a = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp_b = b[:, start:end + 1, :].transpose(1, 0, 2)
    arg_b = np.argsort(tmp_b, axis=0)
    if num == 1:
        pos = end - start
    elif num == 0:
        pos = 0
    else:
        pos = int(num * (end + 1 - start))

    s = np.zeros(tmp_a[0].shape)
    for j in range(pos, end+1-start):
        se = arg_b[j]  # 对应的a的quantile位
        for i in range(end + 1 - start):
            s[se == i] += tmp_a[i][se == i]
    s /= (end+1-start-pos)
    return s


def biintraquantiledownmean(a, b, start, end, num):
    # 根据b排序的a日内分位数下行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    tmp_a = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp_b = b[:, start:end + 1, :].transpose(1, 0, 2)
    arg_b = np.argsort(tmp_b, axis=0)
    if num == 1:
        pos = end - start
    elif num == 0:
        pos = 0
    else:
        pos = int(num * (end + 1 - start))

    s = np.zeros(tmp_a[0].shape)
    for j in range(pos + 1):
        se = arg_b[j]  # 对应的a的quantile位
        for i in range(end + 1 - start):
            s[se == i] += tmp_a[i][se == i]
    s /= (pos + 1)
    return s

