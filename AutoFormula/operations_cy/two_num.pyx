# Copyright (c) 2021 Dai HBG


"""
该代码是2_num_num型运算符cython版本的源代码
"""


import numpy as np
from libc.math cimport isnan, sqrt


def tsregres_2d(double[:, :] a, double [:, :] b, int num):  # 二维回溯num天时序回归残差
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    cdef double avg_a, avg_b, m_2_a, m_2_ab, beta
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            avg_a = 0
            avg_b = 0
            m_2_a = 0
            m_2_ab = 0
            for k in range(i - num + 1, i + 1):
                if (isnan(a[k, j])) or isnan(b[k, j]):
                    break
                avg_a += a[k, j]
                avg_b += b[k, j]
            else:
                avg_a /= num
                avg_b /= num
                for k in range(i - num + 1, i + 1):
                    m_2_a += (a[k, j] - avg_a) ** 2
                    m_2_ab += (a[k, j] - avg_a) * (b[k, j] - avg_b)
                if m_2_a == 0:
                    beta = 0
                else:
                    beta = m_2_ab / m_2_a
                s_view[i, j] = (b[i, j] - avg_b) - beta * (a[i, j] - avg_a)
                continue
            s_view[i, j] = nan
    return s


def tsregres_3d(double[:, :, :] a, double [:, :, :] b, int num):  # 三维回溯num天时序回归残差
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l
    cdef nan = np.nan
    cdef double avg_a, avg_b, m_2_a, m_2_ab, beta
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                avg_a = 0
                avg_b = 0
                m_2_a = 0
                m_2_ab = 0
                for l in range(i - num + 1, i + 1):
                    if (isnan(a[l, j, k])) or (isnan(b[l, j, k])):
                        break
                    avg_a += a[l, j, k]
                    avg_b += b[l, j, k]
                else:
                    avg_a /= num
                    avg_b /= num
                    for l in range(i - num + 1, i + 1):
                        m_2_a += (a[l, j, k] - avg_a) ** 2
                        m_2_ab += (a[l, j, k] - avg_a) * (b[l, j, k] - avg_b)
                    if m_2_a == 0:
                        beta = 0
                    else:
                        beta = m_2_ab / m_2_a
                    s_view[i, j, k] = (b[i, j, k] - avg_b) - beta * (a[i, j, k] - avg_a)
                    continue
                s_view[i, j, k] = nan
    return s


def tscorr_2d(double[:, :] a, double [:, :] b, int num):  # 二维回溯num天时序相关系数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef Py_ssize_t i, j, k
    cdef nan = np.nan
    cdef double avg_a, avg_b, m_2_a, m_2_b, m_2_ab, std
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                s_view[i, j] = nan
            continue
        for j in range(dim_2):
            avg_a = 0
            avg_b = 0
            m_2_a = 0
            m_2_b = 0
            m_2_ab = 0
            for k in range(i - num + 1, i + 1):
                if isnan(a[k, j]) or isnan(b[k, j]):
                    break
                avg_a += a[k, j]
                avg_b += b[k, j]
                m_2_a += a[k, j] * a[k, j]
                m_2_b += b[k, j] * b[k, j]
                m_2_ab += a[k, j] * b[k, j]
            else:  # else是for循环正常执行完后执行的部分
                std = (m_2_a - avg_a * avg_a / num) * (m_2_b - avg_b * avg_b / num)
                if std == 0:
                    s_view[i, j] = 0
                else:
                    s_view[i, j] = (m_2_ab - avg_a * avg_b / num) / sqrt(std)
                continue
            s_view[i, j] = nan
    return s


def tscorr_3d(double[:, :, :] a, double [:, :, :] b, int num):  # 三维回溯num天时序相关系数
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef Py_ssize_t i, j, k, l
    cdef nan = np.nan
    cdef double avg_a, avg_b, m_2_a, m_2_b, m_2_ab, beta
    for i in range(dim_1):
        if i < num - 1:
            for j in range(dim_2):
                for k in range(dim_3):
                    s_view[i, j, k] = nan
            continue
        for j in range(dim_2):
            for k in range(dim_3):
                avg_a = 0
                avg_b = 0
                m_2_a = 0
                m_2_b = 0
                m_2_ab = 0
                for l in range(i - num + 1, i + 1):
                    if (isnan(a[l, j, k])) or isnan(b[l, j, k]):
                        break
                    avg_a += a[l, j, k]
                    avg_b += b[l, j, k]
                else:
                    avg_a /= num
                    avg_b /= num
                    for l in range(i - num + 1, i + 1):
                        m_2_a += (a[l, j, k] - avg_a) ** 2
                        m_2_b += (b[l, j, k] - avg_b) ** 2
                        m_2_ab += (a[l, j, k] - avg_a) * (b[l, j, k] - avg_b)
                    if (m_2_a * m_2_b) == 0:
                        s_view[i, j, k] = 0
                    else:
                        s_view[i, j, k] = m_2_ab / ((m_2_a * m_2_b) ** 0.5)

                    continue
                s_view[i, j, k] = nan
    return s
