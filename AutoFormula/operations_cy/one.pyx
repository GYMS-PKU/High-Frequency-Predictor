# Copyright (c) 2021 Dai HBG


"""
该代码是1型运算符的cython版本源代码，注意需要为不同维度都写操作

日志
2022-01-20
- 新增log操作和logv操作
"""


import numpy as np
from libc.math cimport isnan, log


def neg_2d(double[:, :] a):  # 二维矩阵求相反数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            s_view[i, j] = -a[i, j]
    return s


def neg_3d(double[:, :, :] a):  # 三维矩阵求相反数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                s_view[i, j, k] = -a[i, j, k]

    return s


def absv_2d(double[:, :] a):  # 二维矩阵取绝对值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] < 0:
                s_view[i, j] = -a[i, j]
            else:
                s_view[i, j] = a[i, j]
    return s


def absv_3d(double[:, :, :] a):  # 三维矩阵取绝对值
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
       for j in range(dim_2):
           for k in range(dim_3):
               if isnan(a[i, j, k]):
                   s_view[i, j, k] = nan
                   continue
               if a[i, j, k] < 0:
                   s_view[i, j, k] = -a[i, j, k]
               else:
                   s_view[i, j, k] = a[i, j, k]
    return s


def log_2d(double[:, :] a):  # 二维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] > 0:
                s_view[i, j] = log(a[i, j] + 1e-7)
            else:
                s_view[i, j] = nan
    return s


def log_3d(double[:, :, :] a):  # 三维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] > 0:
                    s_view[i, j, k] = log(a[i, j, k] + 1e-7)
                else:
                    s_view[i, j, k] = nan
    return s


def logv_2d(double[:, :] a):  # 二维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    cdef nan = np.nan
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s_view[i, j] = nan
                continue
            if a[i, j] > 0:
                s_view[i, j] = log(a[i, j] + 1e-7)
            elif a[i, j] < 0:
                s_view[i, j] = -log(-a[i, j] + 1e-7)
            else:
                s_view[i, j] = nan
    return s


def logv_3d(double[:, :, :] a):  # 三维矩阵求log
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2. dim_3))
    cdef double[:, :, :] s_view = s
    cdef nan = np.nan
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s_view[i, j, k] = nan
                    continue
                if a[i, j, k] > 0:
                    s_view[i, j, k] = log(a[i, j, k] + 1e-7)
                elif a[i, j, k] < 0:
                    s_view[i, j, k] = -log(-a[i, j, k] + 1e-7)
            else:
                s_view[i, j, k] = nan
    return s


def intratsfftreal(a):  # 日内fft实数部分
    return np.fft.fft(a, axis=1).real / a.shape[1]  # 归一化


def intratsfftimag(a):  # 日内fft虚数部分
    return np.fft.fft(a, axis=1).imag / a.shape[1]  # 归一化
