# Copyright (c) 2021 Dai HBG


"""
该代码定义1_num_num型运算符
"""


import numpy as np
import numba as nb


# @nb.jit
def intratsmax(a, start, end):  # 返回日内指定时间段的最大值
    return np.max(a[:, start:end + 1, :], axis=1)


# @nb.jit
def intratsmaxpos(a, start, end):  # 返回日内指定时间段的最大值位置
    return np.argmax(a[:, start:end + 1, :], axis=1)


# @nb.jit
def intratsmin(a, start, end):  # 返回日内指定时间段的最小值
    return np.min(a[:, start:end + 1, :], axis=1)


# @nb.jit
def intratsminpos(a, start, end):  # 返回日内指定时间段的最小值位置
    return np.argmin(a[:, start:end + 1, :], axis=1)


# @nb.jit
def intratsstd(a, start, end):  # 日内指定时间段的标准差
    return np.nanstd(a[:, start:end + 1, :], axis=1)


# @nb.jit
def intratsmean(a, start, end):  # 日内指定时间段的均值
    return np.nanmean(a[:, start:end + 1, :], axis=1)


def intratskurtosis(a, start, end):  # 日内峰度
    assert (2 <= (end - start)) and (end < a.shape[1])
    tmp = a[:, start:end + 1, :].transpose(1, 0, 2)
    return (np.mean((tmp - np.mean(tmp, axis=0)) ** 4, axis=0) / np.std(tmp, axis=0) ** 4) - 3


def intratskew(a, start, end):  # 日内偏度
    assert (2 <= (end - start)) and (end < a.shape[1])
    tmp = a[:, start:end + 1, :].transpose(1, 0, 2)
    return np.mean((tmp - np.mean(tmp, axis=0)) ** 3, axis=0) / np.std(tmp, axis=0) ** 3


# @nb.jit
def tsautocorr(a, delta, num):
    s = np.zeros(a.shape)
    tmp_x = np.zeros((num, a.shape[0], a.shape[1]))
    tmp_x[0] = a.copy()
    b = np.zeros(a.shape)
    b[delta:] = a[:-delta].copy()
    tmp_y = np.zeros((num, b.shape[0], b.shape[1]))
    tmp_y[0] = b.copy()
    for i in range(1, num):
        tmp_x[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
        tmp_y[i, i:, :] = b[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = (np.nanmean(tmp_x * tmp_y,
                              axis=0) / (np.nanstd(tmp_x, axis=0) * np.nanstd(tmp_y, axis=0)))[num - 1:]
    return s


# 傅里叶变换类型
def tsfftreal(a, num, pos):  # 时序数据回溯num天做离散fft实数部分的第pos个元素值
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = (np.fft.fft(tmp, axis=0).real[pos] / num)[num - 1:]  # 归一化
    return s


def tsfftimag(a, num, pos):  # 时序数据回溯num天做离散fft虚数部分的第pos个元素值
    assert pos > 0  # 第一位肯定是0，不成为有效信号
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = (np.fft.fft(tmp, axis=0).imag[pos] / num)[num - 1:]  # 归一化
    return s


def tshpf(a, num, pos):  # 时序高通滤波器，将pos之前的置为0
    assert (pos > 0) and (pos < num)
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    f = np.fft.fft(tmp, axis=0)
    f[:pos] = 0
    s[num - 1:] = np.fft.ifft(f, axis=0).real[0][num - 1:]
    s[:num - 1] = np.nan
    return s


def tslpf(a, num, pos):  # 时序低通滤波器，将pos之后的置为0
    assert (pos > 0) and (pos < num)
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    f = np.fft.fft(tmp, axis=0)
    f[pos:] = 0
    s[num - 1:] = np.fft.ifft(f, axis=0).real[0][num - 1:]
    s[:num - 1] = np.nan
    return s


def intratsquantile(a, delay, num):  # 日内时序分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入
    s = np.zeros(a.shape)
    tmp = np.zeros((delay, a.shape[0], a.shape[1], a.shape[2]))
    tmp[0] = a.copy()
    for i in range(1, delay):
        tmp[i, i:, :, :] = a[:-i]  # 第i行存放delay i天的数据
    tmp = np.sort(tmp, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    s[delay - 1:] = tmp[pos][delay - 1:]
    s[:delay - 1] = np.nan
    return s


def tsquantile(a, delay, num):  # 时序分位数算子，num是一个介于0到1之间的数字，这个算子只接受二维输入
    s = np.zeros(a.shape)
    tmp = np.zeros((delay, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, delay):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    tmp = np.sort(tmp, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    s[delay - 1:] = tmp[pos][delay - 1:]
    s[:delay - 1] = np.nan
    return s


def tsquantiledownmean(a, delay, num):  # 时序分位数平均算子，num是一个介于0到1之间的数字，这个算子只接受二维输入
    s = np.zeros(a.shape)  # 返回该排位之下的所有的值的平均
    tmp = np.zeros((delay, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, delay):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    tmp = np.sort(tmp, axis=0)  # 时序排序值
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    s[delay - 1:] = np.mean(tmp[:pos + 1], axis=0)[delay - 1:]
    s[:delay - 1] = np.nan
    return s


def tsquantileupmean(a, delay, num):  # 时序分位数平均算子，num是一个介于0到1之间的数字，这个算子只接受二维输入
    s = np.zeros(a.shape)  # 返回该排位之上的所有的值的平均
    tmp = np.zeros((delay, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, delay):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    tmp = np.sort(tmp, axis=0)  # 时序排序值
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    tmp[tmp_sort < pos] = 0
    s[delay1:] = np.mean(tmp[pos:], axis=0)
    s[:delay - 1] = np.nan
    return s


def intratsquantileupmean(a, delay, num):  # 日内时序分位数上行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    s = np.zeros(a.shape)
    tmp = np.zeros((delay, a.shape[0], a.shape[1], a.shape[2]))
    tmp[0] = a.copy()
    for i in range(1, delay):
        tmp[i, i:, :, :] = a[:-i]  # 第i行存放delay i天的数据
    tmp = np.sort(tmp, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    s[delay - 1:] = np.mean(tmp[pos:], axis=0)[delay - 1:]
    s[:delay - 1] = np.nan
    return s


def intratsquantiledownmean(a, delay, num):  # 日内时序分位数下行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
    s = np.zeros(a.shape)
    tmp = np.zeros((delay, a.shape[0], a.shape[1], a.shape[2]))
    tmp[0] = a.copy()
    for i in range(1, delay):
        tmp[i, i:, :, :] = a[:-i]  # 第i行存放delay i天的数据
    tmp = np.sort(tmp, axis=0)  # 时序排序
    if num == 1:
        pos = delay - 1
    elif num == 0:
        pos = 0
    else:
        pos = int(num * delay)
    s[delay - 1:] = np.mean(tmp[:pos + 1], axis=0)[delay - 1:]  # 时序的需要去掉前num-1天
    s[:delay - 1] = np.nan
    return s
