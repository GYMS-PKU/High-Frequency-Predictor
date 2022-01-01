# Copyright (c) 2021 Dai HBG


"""
该代码定义2_num_num型运算符
"""


import numpy as np
import numba as nb


@nb.jit
def intratscorr(a, b, start, end):  # 日内数据的时序相关性，传入start和end
    assert (2 <= (end - start)) and (end < a.shape[1])
    tmp_a = a[:, start:end + 1, :].transpose(1, 0, 2)
    tmp_b = b[:, start:end + 1, :].transpose(1, 0, 2)
    tmp_a -= np.nanmean(tmp_a, axis=0)
    tmp_b -= np.nanmean(tmp_b, axis=0)
    s = np.zeros((a.shape[0], a.shape[2]))
    s[num - 1:] = (np.nanmean(tmp_a * tmp_b,
                              axis=0) / (np.nanstd(tmp_a, axis=0) * np.nanstd(tmp_b, axis=0)))[num - 1:]
    return s


def biintratsquantile(a, b, delay, num):  # 日内根据b排序的a时序分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入
    s = np.zeros(a.shape)
    tmp_a = np.zeros((delay, a.shape[0], a.shape[1], a.shape[2]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((delay, b.shape[0], b.shape[1], b.shape[2]))
    tmp_b[0] = a.copy()
    for i in range(1, delay):
        tmp_a[i, i:, :, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :, :] = b[:-i]  # 第i行存放delay i天的数据
    arg_b = np.argsort(tmp_b, axis=0)  # b的时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    se = arg_b[pos]
    for i in range(delay):
        # b排名等于pos的对应的是第i个的那些的位置
        s[delay - 1:][se[delay-1:] == i] = tmp_a[i][delay - 1:][se[delay-1:] == i]
    s[:delay - 1] = np.nan
    return s


def bitsquantile(a, b, delay, num):  # 时序分位数算子，num是一个介于0到1之间的数字，这个算子只接受二维输入
    s = np.zeros(a.shape)
    tmp_a = np.zeros((delay, a.shape[0], a.shape[1]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((delay, b.shape[0], b.shape[1]))
    tmp_b[0] = b.copy()
    for i in range(1, delay):
        tmp_a[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :] = b[:-i]
    arg_b = np.argsort(tmp_b, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    se = arg_b[pos]
    for i in range(delay):
        s[delay - 1:][se[delay - 1:] == i] = tmp_a[i][delay - 1:][se[delay - 1:] == i]
    s[:delay - 1] = np.nan
    return s


def bitsquantileupmean(a, delay, num):  # 时序分位数平均算子，num是一个介于0到1之间的数字，这个算子只接受二维输入
    s = np.zeros(a.shape)  # 返回该排位之上的所有的值的平均
    tmp_a = np.zeros((delay, a.shape[0], a.shape[1]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((delay, b.shape[0], b.shape[1]))
    tmp_b[0] = b.copy()
    for i in range(1, delay):
        tmp_a[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :] = b[:-i]
    arg_b = np.argsort(tmp_b, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)

    for j in range(pos, delay):
        se = arg_b[j]  # 对应的a的quantile位
        for i in range(delay):
            s[delay - 1:][se[delay - 1:] == i] += tmp_a[i][delay - 1:][se[delay - 1:] == i]
    s /= (delay - pos)
    s[:delay - 1] = np.nan
    return s


def bitsquantiledownmean(a, b, delay, num):  # 时序分位数平均算子，num是一个介于0到1之间的数字，这个算子只接受二维输入
    s = np.zeros(a.shape)  # 返回该排位之下的所有的值的平均
    tmp_a = np.zeros((delay, a.shape[0], a.shape[1]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((delay, b.shape[0], b.shape[1]))
    tmp_b[0] = b.copy()
    for i in range(1, delay):
        tmp_a[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :] = b[:-i]
    arg_b = np.argsort(tmp_b, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)

    for j in range(pos + 1):
        se = arg_b[j]  # 对应的a的quantile位
        for i in range(delay):
            s[delay - 1:][se[delay - 1:] == i] += tmp_a[i][delay - 1:][se[delay - 1:] == i]
    s /= (pos + 1)
    s[:delay - 1] = np.nan
    return s


def biintratsquantileupmean(a, b, delay, num):  # 日内时序分位数上行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    s = np.zeros(a.shape)
    tmp_a = np.zeros((delay, a.shape[0], a.shape[1], a.shape[2]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((delay, b.shape[0], b.shape[1], b.shape[2]))
    tmp_b[0] = b.copy()
    for i in range(1, delay):
        tmp_a[i, i:, :, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :, :] = b[:-i]
    arg_b = np.sort(tmp_b, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)

    for j in range(pos, delay):
        se = arg_b[j]  # 对应的a的quantile位
        for i in range(delay):
            s[delay - 1:][se[delay - 1:] == i] += tmp_a[i][delay - 1:][se[delay - 1:] == i]
    s /= (delay - pos)
    s[:delay - 1] = np.nan
    return s


def biintratsquantiledownmean(a, b, delay, num):  # 日内时序分位数下行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    s = np.zeros(a.shape)
    tmp_a = np.zeros((delay, a.shape[0], a.shape[1], a.shape[2]))
    tmp_a[0] = a.copy()
    tmp_b = np.zeros((delay, b.shape[0], b.shape[1], b.shape[2]))
    tmp_b[0] = b.copy()
    for i in range(1, delay):
        tmp_a[i, i:, :, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_b[i, i:, :, :] = b[:-i]
    arg_b = np.sort(tmp_b, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)

    for j in range(pos + 1):
        se = arg_b[j]  # 对应的a的quantile位
        for i in range(delay):
            s[delay - 1:][se[delay - 1:] == i] += tmp_a[i][delay - 1:][se[delay - 1:] == i]
    s /= (pos + 1)
    s[:delay - 1] = np.nan
    return s
