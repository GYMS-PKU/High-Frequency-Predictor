# Copyright (c) 2021 Dai HBG


"""
该代码是1_num_num型运算符的Cython版本源代码
"""


import numpy as np
from libc.math cimport isnan, sqrt


def intratsmax_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的最大值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double tmp
    cdef nan = np.nan
    if end < start:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            j = start
            while isnan(a[i, j, k]):
                j += 1
                if j == end + 1:
                    break
            if j == end + 1:  # 日内时段全是nan则填充nan
                s_view[i, k] = nan
            tmp = a[i, j, k]
            j += 1
            while j < end + 1:
                if isnan(a[i, j, k]):
                    j += 1
                    continue
                if a[i, j, k] > tmp:
                    tmp = a[i, j, k]
                j += 1
            s_view[i, k] = tmp
    return s


def intratsmaxpos_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的最大值位置
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double tmp
    cdef int pos
    cdef nan = np.nan
    if end < start:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            j = start
            while isnan(a[i, j, k]):
                j += 1
                if j == end + 1:
                    break
            if j == end + 1:  # 日内时段全是nan则填充nan
                s_view[i, k] = nan
            tmp = a[i, j, k]
            pos = j
            j += 1
            while j < end + 1:
                if isnan(a[i, j, k]):
                    j += 1
                    continue
                if a[i, j, k] > tmp:
                    tmp = a[i, j, k]
                    pos = j
                j += 1
            s_view[i, k] = float(pos)
    return s


def intratsmin_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的最小值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double tmp
    cdef nan = np.nan
    if end < start:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            j = start
            while isnan(a[i, j, k]):
                j += 1
                if j == end + 1:
                    break
            if j == end + 1:  # 日内时段全是nan则填充nan
                s_view[i, k] = nan
            tmp = a[i, j, k]
            j += 1
            while j < end + 1:
                if isnan(a[i, j, k]):
                    j += 1
                    continue
                if a[i, j, k] < tmp:
                    tmp = a[i, j, k]
                j += 1
            s_view[i, k] = tmp
    return s


def intratsminpos_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的最小值位置
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double tmp
    cdef int pos
    cdef nan = np.nan
    if end < start:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            j = start
            while isnan(a[i, j, k]):
                j += 1
                if j == end + 1:
                    break
            if j == end + 1:  # 日内时段全是nan则填充nan
                s_view[i, k] = nan
            tmp = a[i, j, k]
            pos = j
            j += 1
            while j < end + 1:
                if isnan(a[i, j, k]):
                    j += 1
                    continue
                if a[i, j, k] < tmp:
                    tmp = a[i, j, k]
                    pos = j
                j += 1
            s_view[i, k] = float(pos)
    return s


def intratsstd_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的标准差
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double avg, m_2
    cdef nan = np.nan
    if end <= start + 1:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            avg = 0
            m_2 = 0  # 二阶矩
            for j in range(start, end + 1):
                if isnan(a[i, j, k]):
                    s_view[i, k] = nan
                    break
                avg += a[i, j, k]
                m_2+= a[i, j, k] * a[i, j, k]
            else:  # 中途break
                avg /= (end - start + 1)
                m_2 /= (end - start + 1)
                s_view[i, k] = sqrt(m_2 - avg ** 2)
                continue
            s_view[i, k] = nan
    return s


def intratsmean_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的均值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double avg
    cdef nan = np.nan
    if end <= start + 1:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            avg = 0
            for j in range(start, end + 1):
                if isnan(a[i, j, k]):
                    s_view[i, k] = nan
                    break
                avg += a[i, j, k]
            else:  # 中途break
                avg /= (end - start + 1)
                s_view[i, k] = avg
                continue
            s_view[i, k] = nan
    return s


def intratskurtosis_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的峰度
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double avg, m_2, m_3, m_4
    cdef nan = np.nan
    if end <= start + 1:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            avg = 0
            m_2 = 0  # 二阶矩
            m_3 = 0
            m_4 = 0
            for j in range(start, end + 1):
                if isnan(a[i, j, k]):
                    s_view[i, k] = nan
                    break
                avg += a[i, j, k]
                m_2 += a[i, j, k] * a[i, j, k]
                m_3 += a[i, j, k] * a[i, j, k] * a[i, j, k]
                m_4 += a[i, j, k] * a[i, j, k] * a[i, j, k] * a[i, j, k]
            else:  # 中途break
                avg /= (end - start + 1)
                m_2 /= (end - start + 1)
                m_3 /= (end - start + 1)
                m_4 /= (end - start + 1)
                s_view[i, k] = (m_4 - 4 * m_3 * avg + 6 * m_2 * avg * avg - 3 * avg ** 4) / ((m_2 - avg * avg) ** 2) - 3
                continue
            s_view[i, k] = nan
    return s


def intratsskew_3d(double[:, :, :] a, int start, int end):  # 返回日内指定时间段的偏度
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double avg, m_2, m_3
    cdef nan = np.nan
    if end <= start + 1:
        return s
    for i in range(dim_1):
        for k in range(dim_3):
            avg = 0
            m_2 = 0  # 二阶矩
            m_3 = 0
            for j in range(start, end + 1):
                if isnan(a[i, j, k]):
                    s_view[i, k] = nan
                    break
                avg += a[i, j, k]
                m_2 += a[i, j, k] * a[i, j, k]
                m_3 += a[i, j, k] * a[i, j, k] * a[i, j, k]
            else:  # 中途break
                avg /= (end - start + 1)
                m_2 /= (end - start + 1)
                m_3 /= (end - start + 1)
                s_view[i, k] = (m_3 - 3 * m_2 * avg + 2 * avg ** 3) / ((m_2 - avg ** 2) ** 1.5)
                continue
            s_view[i, k] = nan
    return s


def tsautocorr_3d(a, delta, num):
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


# 2d分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入，返回二维，其中排序数组长度为delay + 1
def tsquantile_2d(double[:, :] a, int delay, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    pos = np.zeros(delay + 1).astype(np.int32)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (delay + 1))  # 目标分位值
    if tar == delay + 1:
        tar = delay
    cdef nan = np.nan
    for i in range(dim_1):
        if i < delay:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            pos_view[0] = 0  # 初始化
            if isnan(a[i, j]):  # 只要有一个nan就置为nan
                s_view[i, j] = nan
                continue
            for k in range(1, delay + 1):
                if isnan(a[i - delay + k, j]):  # 只要有一个nan就置为nan
                    s_view[i, j] = nan
                    break
                for tmp_pos in range(k):  # 插入排序
                    if a[i - delay + k, j] < a[i - delay + pos_view[tmp_pos], j]:  # 找到插入位置
                        for tmp_pos_1 in range(k, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = k
                        break
                else:
                    pos_view[k] = k  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
            else:  # 获得了排序
                s_view[i, j] = a[i - delay + pos_view[tar], j]
    return s


# 3d分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入，返回三维，其中排序数组长度为delay + 1
def tsquantile_3d(double[: ,:, :] a, int delay, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, l, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    pos = np.zeros(delay + 1).astype(np.int32)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (delay + 1))  # 目标分位值
    if tar == delay + 1:
        tar = delay
    cdef nan = np.nan
    for i in range(dim_1):
        if i < delay:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                pos_view[0] = 0  # 初始化
                if isnan(a[i, j, k]):  # 只要有一个nan就置为nan
                    s_view[i, j, k] = nan
                    continue
                for l in range(1, delay + 1):
                    if isnan(a[i - delay + l, j, k]):  # 只要有一个nan就置为nan
                        s_view[i, j, k] = nan
                        break
                    for tmp_pos in range(l):  # 插入排序
                        if a[i - delay + l, j, k] < a[i - delay + pos_view[tmp_pos], j, k]:  # 找到插入位置
                            for tmp_pos_1 in range(l, tmp_pos, -1):
                                pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                            pos_view[tmp_pos] = l
                            break
                    else:
                        pos_view[l] = l  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
                else:  # 获得了排序
                    s_view[i, j, k] = a[i - delay + pos_view[tar], j, k]
    return s


def tsquantiledownmean_2d(double[:, :] a, int delay, double num):  # 2维时序下行均值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    pos = np.zeros(delay + 1).astype(np.int32)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (delay + 1))  # 目标分位值
    if tar == delay + 1:
        tar = delay
    cdef double avg
    cdef nan = np.nan
    for i in range(dim_1):
        if i < delay :
            for j in range(dim_2):
                s[i, j] = nan
            continue
        for j in range(dim_2):
            pos_view[0] = 0  # 初始化
            if isnan(a[i, j]):  # 只要有一个nan就置为nan
                s_view[i, j] = nan
                continue
            for k in range(1, delay + 1):
                if isnan(a[i - delay + k, j]):  # 只要有一个nan就置为nan
                    s_view[i, j] = nan
                    break
                for tmp_pos in range(k):  # 插入排序
                    if a[i - delay + k, j] < a[i - delay + pos_view[tmp_pos], j]:  # 找到插入位置
                        for tmp_pos_1 in range(k, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = k
                        break
                else:
                    pos_view[k] = k  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
            else:  # 获得了排序
                avg = 0
                for tmp_pos in range(tar + 1):
                    avg += a[i - delay + pos_view[tmp_pos], j]
                s_view[i, j] = avg / (tar + 1)
    return s

def tsquantiledownmean_3d(double[: ,:, :] a, int delay, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, l, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    pos = np.zeros(delay + 1).astype(np.int32)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (delay + 1))  # 目标分位值
    if tar == delay + 1:
        tar = delay
    cdef double avg
    cdef nan = np.nan
    for i in range(dim_1):
        if i < delay :
            for j in range(dim_2):
                for k in range(dim_3):
                    s[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                pos_view[0] = 0  # 初始化
                if isnan(a[i, j, k]):  # 只要有一个nan就置为nan
                    s_view[i, j, k] = nan
                    continue
                for l in range(1, delay + 1):
                    if isnan(a[i - delay + l, j, k]):  # 只要有一个nan就置为nan
                        s_view[i, j, k] = nan
                        break
                    for tmp_pos in range(l):  # 插入排序
                        if a[i - delay + l, j, k] < a[i - delay + pos_view[tmp_pos], j, k]:  # 找到插入位置
                            for tmp_pos_1 in range(l, tmp_pos, -1):
                                pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                            pos_view[tmp_pos] = l
                            break
                    else:
                        pos_view[l] = l  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
                else:  # 获得了排序
                    avg = 0
                    for tmp_pos in range(tar + 1):
                        avg += a[i - delay + pos_view[tmp_pos], j, k]
                        s_view[i, j, k] = avg / (tar + 1)
    return s


def tsquantileupmean_2d(double[:, :] a, int delay, double num):  # 2维时序上行均值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    pos = np.zeros(delay + 1).astype(np.int32)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (delay + 1))  # 目标分位值
    if tar == delay + 1:
        tar = delay
    cdef double avg
    cdef nan = np.nan
    for i in range(dim_1):
        if i < delay:
            for j in range(dim_2):
                s[i, j] = nan
            continue
        for j in range(dim_2):
            pos_view[0] = 0  # 初始化
            if isnan(a[i, j]):  # 只要有一个nan就置为nan
                s_view[i, j] = nan
                continue
            for k in range(1, delay + 1):
                if isnan(a[i - delay + k, j]):  # 只要有一个nan就置为nan
                    s_view[i, j] = nan
                    break
                for tmp_pos in range(k):  # 插入排序
                    if a[i - delay + k, j] < a[i - delay + pos_view[tmp_pos], j]:  # 找到插入位置
                        for tmp_pos_1 in range(k, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = k
                        break
                else:
                    pos_view[k] = k  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
            else:  # 获得了排序
                avg = 0
                for tmp_pos in range(tar, delay + 1):
                    avg += a[i - delay + pos_view[tmp_pos], j]
                s_view[i, j] = avg / (delay + 1 - tar)
    return s

def tsquantileupmean_3d(double[: ,:, :] a, int delay, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, l, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    pos = np.zeros(delay + 1).astype(np.int32)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (delay + 1))  # 目标分位值
    if tar == delay + 1:
        tar = delay
    cdef double avg
    cdef nan = np.nan
    for i in range(dim_1):
        if i < delay:
            for j in range(dim_2):
                for k in range(dim_3):
                    s[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                pos_view[0] = 0  # 初始化
                if isnan(a[i, j, k]):  # 只要有一个nan就置为nan
                    s_view[i, j, k] = nan
                    continue
                for l in range(1, delay + 1):
                    if isnan(a[i - delay + l, j, k]):  # 只要有一个nan就置为nan
                        s_view[i, j, k] = nan
                        break
                    for tmp_pos in range(l):  # 插入排序
                        if a[i - delay + l, j, k] < a[i - delay + pos_view[tmp_pos], j, k]:  # 找到插入位置
                            for tmp_pos_1 in range(l, tmp_pos, -1):
                                pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                            pos_view[tmp_pos] = l
                            break
                    else:
                        pos_view[l] = l  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
                else:  # 获得了排序
                    avg = 0
                    for tmp_pos in range(tar, delay + 1):
                        avg += a[i - delay + pos_view[tmp_pos], j, k]
                        s_view[i, j, k] = avg / (delay + 1 - tar)
    return s
