# Copyright (c) 2022 Dai HBG


"""
该代码是2_num_num_num型运算符源代码

日志
2022-01-06
- init
"""


import numpy as np
from libc.math cimport isnan


# 二维时序选择算子，回溯delay天，根据b的排序选出从num_1到num_2的a的子集，返回均值，是闭区间
def tssubset_2d(double[:, :] a, double[:, :] b, int delay, int num_1, int num_2):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef double avg
    pos = np.zeros(delay + 1).astype(int)  # 存放b的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
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
                for tmp_pos in range(num_1, num_2 + 1):
                    avg += a[i - delay + 1 + pos_view[tmp_pos], k]
                s_view[i, j] = avg / (num_2 - num_1 + 1)
    return s


# 三维时序选择算子，回溯delay天，根据b的排序选出从num_1到num_2的a的子集，返回均值，是闭区间
def tssubset_3d(double[:, :, :] a, double[:, :, :] b, int delay, int num_1, int num_2):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, l, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef double avg
    pos = np.zeros(delay + 1).astype(int)  # 存放b的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
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
                    for tmp_pos in range(num_1, num_2 + 1):
                        avg += a[i - delay + 1 + pos_view[tmp_pos], j, k]
                    s_view[i, j, k] = avg / (num_2 - num_1 + 1)
    return s


# 根据b排序的a的日内分位数算子，num是一个介于0到1之间的数字，该算子接受三维输入，返回2维矩阵
def biintraquantile_3d(double[:, :, :] a, double[:, :, :] b, int start, int end, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    pos = np.zeros(end - start + 1).astype(int)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (end - start + 1))  # 目标分位值
    cdef nan = np.nan
    if tar == end - start + 1:
        tar -= 1
    if end < start:  # 长度至少是1
        return s

    for i in range(dim_1):
        for k in range(dim_3):
            pos_view[0] = 0  # 初始化
            if isnan(a[i, start, k]) or isnan(b[i, start, k]):  # 只要有一个nan就置为nan
                s_view[i, k] = nan
                continue
            for j in range(start + 1, end + 1):
                if isnan(a[i, j, k]) or isnan(b[i, j, k]):  # 只要有一个nan就置为nan
                    s_view[i, k] = nan
                    break
                for tmp_pos in range(j - start):  # 插入排序
                    if b[i, j, k] < b[i, pos_view[tmp_pos] + start, k]:  # 比排在tmp_pos位置的元素小，于是tmp_pos之后的平移
                        for tmp_pos_1 in range(j-start, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = j - start  # 插入
                        break
                else:  # 此时a[i, j, k]是最大的
                    pos_view[j - start] = j - start
                    continue
            else:  # 获得了排序
                s_view[i, k] = a[i, start + pos_view[tar], k]
    return s


# 根据b排序的a日内分位数上行平均算子，num是一个介于0到1之间的数字，该算子接受三维输入
def biintraquantileupmean_3d(double[:, :, :] a, double[:, :, :] b, int start, int end, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    cdef double avg   # 平均值
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    pos = np.zeros(end - start + 1).astype(int)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (end - start + 1))  # 目标分位值
    cdef nan = np.nan
    if tar == end - start + 1:
        tar -= 1
    if end < start:  # 长度至少是1
        return s

    for i in range(dim_1):
        for k in range(dim_3):
            pos_view[0] = 0  # 初始化
            if isnan(a[i, start, k]) or isnan(b[i, start, k]):  # 只要有一个nan就置为nan
                s_view[i, k] = nan
                continue
            for j in range(start + 1, end + 1):
                if isnan(a[i, j, k]) or isnan(a[i, j ,k]):  # 只要有一个nan就置为nan
                    s_view[i, k] = nan
                    break
                for tmp_pos in range(j - start):  # 插入排序
                    if b[i, j, k] < b[i, pos_view[tmp_pos] + start, k]:  # 比排在tmp_pos位置的元素小，于是tmp_pos之后的平移
                        for tmp_pos_1 in range(j-start, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = j - start  # 插入
                        break
                else:  # 此时a[i, j, k]是最大的
                    pos_view[j - start] = j - start
                    continue
            else:  # 获得了排序
                avg = 0
                for tmp_pos in range(start + tar, end + 1):
                    avg += a[i, start + pos_view[tmp_pos], k]
                s_view[i, k] = avg / (end - start + 1 - tar)
    return s


def biintraquantiledownmean_3d(double[:, :, :] a, double[:, :, :] b, int start, int end, double num):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k, tmp_pos, tmp_pos_1
    cdef double avg   # 平均值
    s = np.zeros((dim_1, dim_3))
    cdef double[:, :] s_view = s
    pos = np.zeros(end - start + 1).astype(int)  # 存放a的时序排序值，pos[i]存放对应array中从小到大排第i的元素的位置
    cdef int[:] pos_view = pos
    cdef int tar = int(num * (end - start + 1))  # 目标分位值
    cdef nan = np.nan
    if tar == end - start + 1:
        tar -= 1
    if end < start:  # 长度至少是1
        return s

    for i in range(dim_1):
        for k in range(dim_3):
            pos_view[0] = 0  # 初始化
            if isnan(a[i, start, k]) or isnan(b[i, start, k]):  # 只要有一个nan就置为nan
                s_view[i, k] = nan
                continue
            for j in range(start + 1, end + 1):
                if isnan(a[i, j, k]) or isnan(a[i, j ,k]):  # 只要有一个nan就置为nan
                    s_view[i, k] = nan
                    break
                for tmp_pos in range(j - start):  # 插入排序
                    if b[i, j, k] < b[i, pos_view[tmp_pos] + start, k]:  # 比排在tmp_pos位置的元素小，于是tmp_pos之后的平移
                        for tmp_pos_1 in range(j-start, tmp_pos, -1):
                            pos_view[tmp_pos_1] = pos_view[tmp_pos_1 - 1]  # 平移
                        pos_view[tmp_pos] = j - start  # 插入
                        break
                else:  # 此时a[i, j, k]是最大的
                    pos_view[j - start] = j - start
                    continue
            else:  # 获得了排序
                avg = 0
                for tmp_pos in range(start, start + tar + 1):
                    avg += a[i, start + pos_view[tmp_pos], k]
                s_view[i, k] = avg / (tar + 1)
    return s
