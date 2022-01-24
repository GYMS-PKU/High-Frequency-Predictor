# Copyright (c) 2021 Dai HBG


"""
该代码是2型运算符Cython版本的源代码
"""


import numpy as np
from libc.math cimport isnan


def add_2d(double[:, :] a, double[:, :] b):  # 二维矩阵加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] + b[i, j]
    return s


def add_num_2d(double[:, :] a, double b):  # 二维矩阵数字加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] + b
    return s


def add_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] + b[i, j, k]
    return s


def add_num_3d(double[:, :, :] a, double b):  # 三维矩阵数字加法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] + b
    return s


def minus_2d(double[:, :] a, double[:, :] b):  # 二维矩阵减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] - b[i, j]
    return s


def minus_num_2d(double[:, :] a, double b):  # 二维矩阵数字减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] - b
    return s


def minus_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] - b[i, j, k]
    return s


def minus_num_3d(double[:, :, :] a, double b):  # 三维矩阵数字减法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] - b
    return s


def prod_2d(double[:, :] a, double[:, :] b):  # 二维矩阵乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] * b[i, j]
    return s


def prod_num_2d(double[:, :] a, double b):  # 二维矩阵数字乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] * b
    return s


def prod_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] * b[i, j, k]
    return s


def prod_num_3d(double[:, :, :] a, double b):  # 三维矩阵数字乘法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] * b
    return s


def div_2d(double[:, :] a, double[:, :] b):  # 二维矩阵除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]) or (b[i, j] == 0):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] / b[i, j]
    return s


def div_num_2d(double[:, :] a, double b):  # 二维矩阵数字除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]) or (b == 0):
                s_view[i, j] = nan
                continue
            s_view[i, j] = a[i, j] / b
    return s


def div_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]) or (b[i, j, k] == 0):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] / b[i, j, k]
    return s


def div_num_3d(double[:, :, :] a, double b):  # 三维矩阵数字除法
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]) or (b == 0):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = a[i, j, k] / b
    return s


def intratsregres_3d(double[:, :, :] a, double[:, :, :] b):  # 日内时序回归残差
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef double avg_a, avg_b, m_2_a, m_2_ab, beta
    cdef nan = np.nan
    for i in range(dim_1):
        for k in range(dim_3):
            avg_a = 0
            avg_b = 0
            m_2_a = 0
            m_2_ab = 0
            for j in range(dim_2):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    break
                avg_a += a[i, j, k]
                avg_b += b[i, j, k]
            else:
                avg_a /= dim_2
                avg_b /= dim_2
                for j in range(dim_2):
                    m_2_a += (a[i, j, k] - avg_a) ** 2
                    m_2_ab += (a[i, j, k] - avg_a) * (b[i, j, k] - avg_b)
                if m_2_a == 0:
                    beta = 0
                else:
                    beta = m_2_ab / m_2_a
                for j in range(dim_2):
                    s_view[i, j, k] = (b[i, j, k] - avg_b) - beta * (a[i, j, k] - avg_a)
                continue
            for j in range(dim_2):
                s_view[i, j, k] = nan
    return s


def lt_2d(double[:, :] a, double[:, :] b):  # 二维矩阵比较小于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] < b[i, j]:
                s_view[i, j] = 1
            else:
                s_view[i, j] = 0
    return s


def lt_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵比较小于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] < b[i, j, k]:
                    s_view[i, j, k] = 1
                else:
                    s_view[i, j, k] = 0
    return s


def le_2d(double[:, :] a, double[:, :] b):  # 二维矩阵比较小于等于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] <= b[i, j]:
                s_view[i, j] = 1
            else:
                s_view[i, j] = 0
    return s


def le_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵比较小于等于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] <= b[i, j, k]:
                    s_view[i, j, k] = 1
                else:
                    s_view[i, j, k] = 0
    return s


def gt_2d(double[:, :] a, double[:, :] b):  # 二维矩阵比较大于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] > b[i, j]:
                s_view[i, j] = 1
            else:
                s_view[i, j] = 0
    return s


def gt_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵比较大于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] > b[i, j, k]:
                    s_view[i, j, k] = 1
                else:
                    s_view[i, j, k] = 0
    return s


def ge_2d(double[:, :] a, double[:, :] b):  # 二维矩阵比较大于等于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            if (isnan(a[i, j])) or isnan(b[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] >= b[i, j]:
                s_view[i, j] = 1
            else:
                s_view[i, j] = 0
    return s


def ge_3d(double[:, :, :] a, double[:, :, :] b):  # 三维矩阵比较大于等于
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if (isnan(a[i, j, k])) or isnan(b[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] >= b[i, j, k]:
                    s_view[i, j, k] = 1
                else:
                    s_view[i, j, k] = 0
    return s
