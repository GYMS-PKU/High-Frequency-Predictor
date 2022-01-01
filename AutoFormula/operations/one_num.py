# Copyright (c) 2021 Dai HBG


"""
该代码定义1_num型运算符


"""


import numpy as np
import numba as nb


def powv(a: np.array, num: float) -> np.array:  # 幂函数运算符
    s = a.copy()
    s[np.isnan(a)] = 0
    s[(s > 0) & (~np.isnan(a))] = a[(s > 0) & (~np.isnan(a))] ** num
    s[(s < 0) & (~np.isnan(a))] = -((-a[(s < 0) & (~np.isnan(a))]) ** num)
    return s


# @nb.jit
def tsmax(a, num):  # 返回最大值
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    notnannum = np.sum(~np.isnan(tmp), axis=0)  # 统计该位置回溯若干天中非nan的个数，为0的需要单独置为nan
    tmp[:, notnannum == 0] = 0  # 将全部是nan的地方先置为0
    tmp_max = np.nanmax(tmp, axis=0)  # 求最大值
    tmp_max[notnannum == 0] = np.nan  # 这些地方本来应该是nan
    s[num - 1:] = tmp_max[num - 1:]
    s[:num - 1] = np.nan  # 没有值的地方必须用nan填充
    return s


# @nb.jit
def tsmaxpos(a, num):  # 返回最大值位置
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = np.argmax(tmp, axis=0)[num - 1:]
    return s


# @nb.jit
def tsmin(a, num):  # 返回最小值
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = np.min(tmp, axis=0)[num - 1:]
    return s


# @nb.jit
def tsminpos(a, num):  # 返回最小值位置
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = np.argmin(tmp, axis=0)[num - 1:]
    return s


def tsdelay(a, num):
    s = np.zeros(a.shape)
    s[num:] = a[:-num].copy()
    return s


def tsdelta(a, num):  # 数据变化
    s = np.zeros(a.shape)
    s[num:] = a[num:] - a[:-num]
    return s


# @nb.jit
def tsstd(a, num):
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = np.nanstd(tmp, axis=0)[num - 1:]
    return s


def tspct(a, num):  # 数据变化率
    s = np.zeros(a.shape)
    s[num:][a[:-num] != 0] = a[num:][a[:-num] != 0] / a[:-num][a[:-num] != 0] - 1
    s[:num] = np.nan
    return s


# @nb.jit
def tsmean(a, num):
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = np.nanmean(tmp, axis=0)[num - 1:]
    return s


# @nb.jit
def tskurtosis(a, num):
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = (np.mean((tmp - np.mean(tmp, axis=0)) ** 4, axis=0) / np.std(tmp, axis=0) ** 4)[num - 1:] - 3
    return s


# @nb.jit
def tsskew(a, num):  # 时序偏度
    s = np.zeros(a.shape)
    tmp = np.zeros((num, a.shape[0], a.shape[1]))
    tmp[0] = a.copy()
    for i in range(1, num):
        tmp[i, i:, :] = a[:-i]  # 第i行存放delay i天的数据
    s[num - 1:] = (np.mean((tmp - np.mean(tmp, axis=0)) ** 3, axis=0) / np.std(tmp, axis=0) ** 3)[num - 1:]
    return s


# @nb.jit
def wdirect(a, num):  # 过去一段时间中心化之后时序加权
    s = np.zeros(a.shape)
    w = np.array([i for i in range(1, num + 1)])
    for i in range(len(a)):
        if i < num - 1:
            continue
        for j in range(a.shape[1]):
            if np.std(a[i - num + 1:i + 1, j]) == 0:
                continue
            s[i, j] = np.sum(w * (a[i - num + 1:i + 1, j] - np.mean(a[i - num + 1:i + 1,
                                                                    j]))) / np.std(a[i - num + 1:i + 1, j])
    return s


# @nb.jit
def tsrank(a, num):
    s = np.zeros(a.shape)
    for i in range(len(a)):
        if i < num - 1:
            continue
        for j in range(a.shape[1]):
            k = 0
            tar = a[i, j]
            if np.isnan(tar):
                tar = 0
            for c in a[i - num + 1:i + 1, j]:
                if np.isnan(c):
                    continue
                if c < tar:
                    k += 1
            s[i, j] = k / (num - 1)
    return s


def intratslpf(a, pos):  # 日内时序低通滤波器，将pos之前的置为0
    assert (pos > 0) and (pos < a.shape[1])
    f = np.fft.fft(a, axis=1)
    f[:, pos:, :] = 0
    return np.fft.ifft(f, axis=1).real


def intratshpf(a, pos):  # 日内时序高通滤波器，将pos之前的置为0
    assert (pos > 0) and (pos < a.shape[1])
    f = np.fft.fft(a, axis=1)
    f[:, :pos, :] = 0
    return np.fft.ifft(f, axis=1).real
