# Copyright (c) 2021 Dai HBG

"""
该代码是1_num型运算符Cython版本的源代码，需要写2D和3D的版本

说明：
- 时序最值位置默认从起始位置算起，例如如果当前是回溯10天最大值，则tsmaxpos(data, 10) = 9

日志
2021-12-25
- init
2021-01-20
- 新增截断算子
"""

import numpy as np
from libc.math cimport isnan, sqrt


def powv_2d(double[:, :] a, double num):  # 二维幂函数运算符
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] >= 0:
                s_view[i, j] = a[i, j] ** num
            else:
                s_view[i, j] = -((-a[i, j]) ** num)
    return s


def powv_3d(double[:, :, :] a, double num):  # 三维幂函数运算符
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] >= 0:
                    s_view[i, j, k] = a[i, j, k] ** num
                else:
                    s_view[i, j, k] = -((-a[i, j, k]) ** num)
    return s


def tsmax_2d(double[:, :] a, int num):  # 二维求时序最大值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            k = 0
            while isnan(a[i-num+1+k, j]):  # 找到第一个不是nan的
                k += 1
                if k == num:
                    break
            if k == num:  # 全是nan
                s_view[i, j] = nan
                continue
            tmp = a[i-num+1+k, j]  # 遍历记录最大值
            k += 1
            while k < num:
                if isnan(a[i-num+1+k, j]):
                    k += 1
                    continue
                if a[i-num+1+k, j] > tmp:
                    tmp = a[i-num+1+k, j]
                k += 1
            s_view[i, j] = tmp
    return s


def tsmax_3d(double[:, :, :] a, int num):  # 三维求时序最大值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if i < num - 1:
                for k in range(dim_3):
                    s_view[i, j, k] = nan
                continue
            for k in range(dim_3):
                l = 0
                while isnan(a[i-num+1+l, j, k]):  # 找到第一个不是nan的
                    l += 1
                    if l == num:
                        break
                if l == num:  # 全是nan
                    s_view[i, j, k] = nan
                    continue
                tmp = a[i-num+1+l, j, k]  # 遍历记录最大值
                l += 1
                while l < num:
                    if isnan(a[i-num+1+l, j, k]):
                        l += 1
                        continue
                    if a[i-num+1+l, j, k] > tmp:
                        tmp = a[i-num+1+l, j, k]
                    l += 1
                s_view[i, j, k] = tmp
    return s


def tsmaxpos_2d(double[:, :] a, int num):  # 2D计算时序最大值位置
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k, pos
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            k = 0
            while isnan(a[i - num + 1 + k, j]):  # 找到第一个不是nan的
                k += 1
                if k == num:
                    break
            if k == num:  # 全是nan
                s_view[i, j] = nan
                continue
            pos = k  # 遍历记录最大值位置
            tmp = a[i - num + 1 + k, j]
            k += 1
            while k < num:
                if isnan(a[i - num + 1 + k, j]):
                    k += 1
                    continue
                if a[i - num + 1 + k, j] > tmp:
                    pos = k
                    tmp = a[i - num + 1 + k, j]
                k += 1
            s_view[i, j] = float(pos)  # 存储浮点数的位置
    return s


def tsmaxpos_3d(double[:, :, :] a, int num):  # 三维求时序最大值位置
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l, pos
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                l = 0
                while isnan(a[i-num+1+l, j, k]):  # 找到第一个不是nan的
                    l += 1
                    if l == num:
                        break
                if l == num:  # 全是nan
                    s_view[i, j, k] = nan
                    continue
                tmp = a[i-num+1+l, j, k]  # 遍历记录最大值
                pos = l  # 记录最大值位置
                l += 1
                while l < num:
                    if isnan(a[i-num+1+l, j, k]):
                        l += 1
                        continue
                    if a[i-num+1+l, j, k] > tmp:
                        tmp = a[i-num+1+l, j, k]
                        pos = l
                    l += 1
                s_view[i, j, k] = float(pos)
    return s


def tsmin_2d(double[:, :] a, int num):  # 二维求时序最小值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            k = 0
            while isnan(a[i-num+1+k, j]):  # 找到第一个不是nan的
                k += 1
                if k == num:
                    break
            if k == num:  # 全是nan
                s_view[i, j] = nan
                continue
            tmp = a[i-num+1+k, j]  # 遍历记录最大值
            k += 1
            while k < num:
                if isnan(a[i-num+1+k, j]):
                    k += 1
                    continue
                if a[i-num+1+k, j] < tmp:
                    tmp = a[i-num+1+k, j]
                k += 1
            s_view[i, j] = tmp
    return s


def tsmin_3d(double[:, :, :] a, int num):  # 三维求时序最小值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if i < num - 1:
                for k in range(dim_3):
                    s_view[i, j, k] = nan
                continue
            for k in range(dim_3):
                l = 0
                while isnan(a[i-num+1+l, j, k]):  # 找到第一个不是nan的
                    l += 1
                    if l == num:
                        break
                if l == num:  # 全是nan
                    s_view[i, j, k] = nan
                    continue
                tmp = a[i-num+1+l, j, k]  # 遍历记录最大值
                l += 1
                while l < num:
                    if isnan(a[i-num+1+l, j, k]):
                        l += 1
                        continue
                    if a[i-num+1+l, j, k] < tmp:
                        tmp = a[i-num+1+l, j, k]
                    l += 1
                s_view[i, j, k] = tmp
    return s


def tsminpos_2d(double[:, :] a, int num):  # 2D计算时序最小值位置
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k, pos
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            k = 0
            while isnan(a[i - num + 1 + k, j]):  # 找到第一个不是nan的
                k += 1
                if k == num:
                    break
            if k == num:  # 全是nan
                s_view[i, j] = nan
                continue
            pos = k  # 遍历记录最大值位置
            tmp = a[i - num + 1 + k, j]
            k += 1
            while k < num:
                if isnan(a[i - num + 1 + k, j]):
                    k += 1
                    continue
                if a[i - num + 1 + k, j] < tmp:
                    pos = k
                    tmp = a[i - num + 1 + k, j]
                k += 1
            s_view[i, j] = float(pos)  # 存储浮点数的位置
    return s


def tsminpos_3d(double[:, :, :] a, int num):  # 三维求时序最小值位置
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l, pos
    cdef double tmp
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if i < num - 1:
                for k in range(dim_3):
                    s_view[i, j, k] = nan
                continue
            for k in range(dim_3):
                l = 0
                while isnan(a[i-num+1+l, j, k]):  # 找到第一个不是nan的
                    l += 1
                    if l == num:
                        break
                if l == num:  # 全是nan
                    s_view[i, j, k] = nan
                    continue
                tmp = a[i-num+1+l, j, k]  # 遍历记录最大值
                pos = l  # 记录最大值位置
                l += 1
                while l < num:
                    if isnan(a[i-num+1+l, j, k]):
                        l += 1
                        continue
                    if a[i-num+1+l, j, k] < tmp:
                        tmp = a[i-num+1+l, j, k]
                        pos = l
                    l += 1
                s_view[i, j, k] = float(pos)
    return s


def tsdelay_2d(double[:, :] a, int num):  # 二维时序delay
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if i < num:
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i - num, j]
    return s


def tsdelay_3d(double[:, :, :] a, int num):  # 三维时序delay
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s =  np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                s_view[i, j, k] = a[i - num, j, k]
    return s


def tsdelta_2d(double[:, :] a, int num):  # 二维计算时序变化
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            if (isnan(a[i - num, j])) or (isnan(a[i, j])):
                s_view[i ,j] = nan
                continue
            s_view[i, j] = a[i, j] - a[i - num, j]
    return s


def tsdelta_3d(double[:, :, :] a, int num):  # 三维计算时序变化
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s =  np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i - num, j, k])) or (isnan(a[i, j, k])):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] - a[i - num, j, k]
    return s


def tspct_2d(double[:, :] a, int num):  # 二维计算时序变化率
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            if (isnan(a[i - num, j])) or (isnan(a[i, j])) or (a[i - num, j] == 0):
                s_view[i ,j] = nan
                continue
            s_view[i, j] = a[i, j] / a[i - num, j] - 1
    return s


def tspct_3d(double[:, :, :] a, int num):  # 三维计算时序变化率
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i - num, j, k])) or (isnan(a[i, j, k])) or (a[i - num, j, k] == 0):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] / a[i - num, j, k] - 1
    return s


def tsmean_2d(double[:, :] a, int num):  # 二维计算时序均值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double avg
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            avg = 0
            for k in range(num):
                if isnan(a[i - num + 1 + k, j]):
                    s_view[i, j] = nan
                    break
                avg += a[i - num + 1 + k, j]
            else:
                avg /= num
                s_view[i, j] = avg
                continue
            s_view[i, j] = nan
    return s


def tsmean_3d(double[:, :, :] a, int num):  # 三维计算时序均值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l
    cdef double avg
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                avg = 0
                for l in range(num):
                    if isnan(a[i - num + 1 + l, j, k]):
                        s_view[i, j, k] = nan
                        break
                    avg += a[i - num + 1 + l, j, k]
                else:
                    avg /= num
                    s_view[i, j, k] = avg
                    continue
                s_view[i, j, k] = nan
    return s


def tsstd_2d(double[:, :] a, int num):  # 二维计算时序标准差
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef int i, j, k
    cdef double avg, m_2
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            avg = 0
            m_2 = 0  # 二阶矩
            for k in range(num):
                if isnan(a[i - num + 1 + k, j]):
                    s_view[i, j] = nan
                    break
                avg += a[i - num + 1 + k, j]
                m_2 += a[i - num + 1 + k, j] * a[i - num + 1 + k, j]
            else:
                avg /= num
                m_2 /= num
                s_view[i, j] = sqrt(m_2 - avg * avg)
                continue
            s_view[i, j] = nan
    return s


def tsstd_3d(double[:, :, :] a, int num):  # 三维计算时序标准差
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef int i, j, k, l
    cdef double avg, m_2
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                avg = 0
                m_2 = 0
                for l in range(num):
                    if isnan(a[i - num + 1 + l, j, k]):
                        s_view[i, j, k] = nan
                        break
                    avg += a[i - num + 1 + l, j, k]
                    m_2 += a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k]
                else:
                    avg /= num
                    m_2 /= num
                    s_view[i, j, k] = sqrt(m_2 - avg * avg)
                    continue
                s_view[i, j, k] = nan
    return s


def tskurtosis_2d(double[:, :] a, int num):  # 计算二维时序峰度
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef int i, j, k
    cdef double avg, m_2, m_3, m_4
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            avg = 0
            m_2 = 0  # 二阶矩
            m_3 = 0  # 三阶矩
            m_4 = 0  # 四阶矩
            for k in range(num):
                if isnan(a[i - num + 1 + k, j]):
                    s_view[i, j] = nan
                    break
                avg += a[i - num + 1 + k, j]
                m_2 += a[i - num + 1 + k, j] * a[i - num + 1 + k, j]
                m_3 += a[i - num + 1 + k, j] * a[i - num + 1 + k, j] * a[i - num + 1 + k, j]
                m_4 += a[i - num + 1 + k, j] * a[i - num + 1 + k, j] * a[i - num + 1 + k, j] * a[i - num + 1 + k, j]
            else:
                avg /= num
                m_2 /= num
                m_3 /= num
                m_4 /= num
                if (m_2 - avg ** 2) ** 2  > 0:
                    s_view[i, j] = (m_4 - 4 * m_3 * avg + 6 * m_2 * avg ** 2 - 3 * avg ** 4) / \
                                   ((m_2 - avg ** 2) ** 2) - 3
                else:
                    s_view[i, j] = nan
                continue
            s_view[i, j] = nan
    return s


def tskurtosis_3d(double[:, :, :] a, int num):  # 三维计算时序峰度
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef int i, j, k, l
    cdef double avg, m_2, m_3, m_4
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                avg = 0
                m_2 = 0
                m_3 = 0
                m_4 = 0
                for l in range(num):
                    if isnan(a[i - num + 1 + l, j, k]):
                        s_view[i, j, k] = nan
                        break
                    avg += a[i - num + 1 + l, j, k]
                    m_2 += a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k]
                    m_3 += a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k] *  a[i - num + 1 + l, j, k]
                    m_4 += a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k] * \
                           a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k]
                else:
                    avg /= num
                    m_2 /= num
                    m_3 /= num
                    m_4 /= num
                    if (m_2 - avg ** 2) ** 2 > 0:
                        s_view[i, j, k] = (m_4 - 4 * m_3 * avg + 6 * m_2 * avg ** 2 - 3 * avg ** 4) / \
                                          ((m_2 - avg ** 2) ** 2) - 3
                    else:
                        s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = nan
    return s


def tsskew_2d(double[:, :] a, int num):  # 计算二维时序偏度
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s =  np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef int i, j, k
    cdef double avg, m_2, m_3
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            avg = 0
            m_2 = 0  # 二阶矩
            m_3 = 0  # 三阶矩
            for k in range(num):
                if isnan(a[i - num + 1 + k, j]):
                    s_view[i, j] = nan
                    break
                avg += a[i - num + 1 + k, j]
                m_2 += a[i - num + 1 + k, j] * a[i - num + 1 + k, j]
                m_3 += a[i - num + 1 + k, j] * a[i - num + 1 + k, j] * a[i - num + 1 + k, j]
            else:
                avg /= num
                m_2 /= num
                m_3 /= num
                if (m_2 - avg ** 2) ** 1.5 > 0:
                    s_view[i, j] = (m_3 - 3 * m_2 * avg + 2 * avg ** 3) / ((m_2 - avg ** 2) ** 1.5)
                else:
                    s_view[i, j] = nan
                continue
            s_view[i, j] = nan
    return s


def tsskew_3d(double[:, :, :] a, int num):  # 三维计算时序偏度
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef int i, j, k, l
    cdef double avg, m_2, m_3
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                avg = 0
                m_2 = 0
                m_3 = 0
                for l in range(num):
                    if isnan(a[i - num + 1 + l, j, k]):
                        s_view[i, j, k] = nan
                        break
                    avg += a[i - num + 1 + l, j, k]
                    m_2 += a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k]
                    m_3 += a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k] * a[i - num + 1 + l, j, k]
                else:
                    avg /= num
                    m_2 /= num
                    m_3 /= num
                    if (m_2 - avg ** 2) ** 1.5 > 0:
                        s_view[i, j, k] = (m_3 - 3 * m_2 * avg + 2 * avg ** 3) / ((m_2 - avg ** 2) ** 1.5)
                    else:
                        s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = nan
    return s


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


def tsrank_2d(double[:, :] a, int num):  # 二维时序计算排序
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k, pos
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            pos = 0
            for k in range(num):
                if isnan(a[i - num + 1 + k, j]):
                    s_view[i, j] = nan
                    break
                if a[i - num + 1 + k, j] < a[i, j]:
                    pos += 1
            else:
                s_view[i, j] = float(pos) / (num - 1)  # 存储浮点数的位置
                continue
            s_view[i, j] = nan
    return s


def tsrank_3d(double[:, :, :] a, int num):  # 三维时序计算排序
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l, pos
    cdef nan = np.nan
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                pos = 0
                for l in range(num):
                    if isnan(a[i - num + 1 + l, j, k]):
                        s_view[i, j, k] = nan
                        break
                    if a[i - num + 1 + l, j, k] < a[i, j, k]:
                        pos += 1
                else:
                    s_view[i, j, k] = float(pos) / (num - 1)  # 存储浮点数的位置
                    continue
                s_view[i, j, k] = nan
    return s


# def trunc_2d(double[:, :] a, double num):  # 二维截断
#     cdef Py_ssize_t dim_1 = a.shape[0]
#     cdef Py_ssize_t dim_2 = a.shape[1]
#     s = np.zeros((dim_1, dim_2))
#     cdef double[:, :] s_view = s
#     cdef Py_ssize_t i, j, k, pos
#     cdef nan = np.nan
#     for i in range(dim_1):
#         if i < num - 1:
#             for j in range(dim_2):
#                 s_view[i, j] = nan
#             continue
#         for j in range(dim_2):
#             if isnan(a[i, j]):
#                 s_view[i, j] = nan
#                 continue
#             pos = 0
#             for k in range(num):
#                 if isnan(a[i - num + 1 + k, j]):
#                     s_view[i, j] = nan
#                     break
#                 if a[i - num + 1 + k, j] < a[i, j]:
#                     pos += 1
#             else:
#                 s_view[i, j] = float(pos) / (num - 1)  # 存储浮点数的位置
#                 continue
#             s_view[i, j] = nan
#     return s


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
