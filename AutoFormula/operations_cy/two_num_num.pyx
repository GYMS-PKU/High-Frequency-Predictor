# Copyright (c) 2021-2022 Dai HBG


"""
该代码为2_num_num型运算符Cython版本的源代码
"""


import numpy as np
from libc.math cimport isnan, sqrt


def intratscorr_3d(double[: ,:, :] a, double[:, :, :] b, int start, int end):  # 日内数据的时序相关性，传入start和end
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    cdef double avg_a, avg_b, m_2_a, m_2_b, m_2_ab
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for k in range(dim_3):
            avg_a = 0
            avg_b = 0
            m_2_a = 0
            m_2_b = 0
            m_2_ab = 0
            for j in range(dim_2):
                if isnan(a[i, j, k]) or isnan(b[i, j, k]):
                    s_view[i, k] = nan
                    break
                avg_a += a[i, j, k]
                avg_b += b[i, j, k]
                m_2_a += a[i, j, k] * a[i, j, k]
                m_2_b += b[i, j, k] * b[i, j, k]
                m_2_ab += a[i, j, k] * b[i, j, k]
            else:
                avg_a /= (end - start + 1)
                avg_b /= (end - start + 1)
                m_2_a /= (end - start + 1)
                m_2_b /= (end - start + 1)
                m_2_ab /= (end - start + 1)
                std = (m_2_a - avg_a * avg_a) * (m_2_b - avg_b * avg_b)
                if std == 0:
                    s_view[i, k] = 0
                else:
                    s_view[i, k] = (m_2_ab - avg_a * avg_b) / sqrt(std)
    return s


# 二维时序分位数算子，num是一个介于0到1之间的数字
def bitsquantile_2d(double[:, :] a, double[:, :] b, int delay, double num):
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
                    if b[i - delay + k, j] < b[i - delay + pos_view[tmp_pos], j]:  # 找到插入位置
                        for tmp_pos_1 in range(k, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = k
                        break
                else:
                    pos_view[k] = k  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
            else:  # 获得了排序
                s_view[i, j] = a[i - delay + pos_view[tar], j]
    return s


# 三维时序排序分位数
def bitsquantile_3d(double[: ,:, :] a, double[:, :, :] b, int delay, double num):
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
                        if b[i - delay + l, j, k] < b[i - delay + pos_view[tmp_pos], j, k]:  # 找到插入位置
                            for tmp_pos_1 in range(l, tmp_pos, -1):
                                pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                            pos_view[tmp_pos] = l
                            break
                    else:
                        pos_view[l] = l  # 此时a[i - delay + 1 + l, j, k]是前l个中最大的
                else:  # 获得了排序
                    s_view[i, j, k] = a[i - delay + pos_view[tar], j, k]
    return s


def bitsquantiledownmean_2d(double[:, :] a, double[:, :] b, int delay, double num):  # 2维时序下行均值
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
                    if b[i - delay + k, j] < b[i - delay + pos_view[tmp_pos], j]:  # 找到插入位置
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


def bitsquantiledownmean_3d(double[: ,:, :] a, double[: ,:, :] b, int delay, double num):  # 三维时序下行均值
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
                        if b[i - delay + l, j, k] < b[i - delay + pos_view[tmp_pos], j, k]:  # 找到插入位置
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


def bitsquantileupmean_2d(double[:, :] a, double[:, :] b, int delay, double num):  # 2维时序上行均值
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
                    if b[i - delay + k, j] < b[i - delay + pos_view[tmp_pos], j]:  # 找到插入位置
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


def bitsquantileupmean_3d(double[: ,:, :] a, double[: ,:, :] b, int delay, double num):
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
                        if b[i - delay + l, j, k] < b[i - delay + pos_view[tmp_pos], j, k]:  # 找到插入位置
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
