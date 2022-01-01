# Copyright (c) 2021 Dai HBG


"""
该代码定义1型运算符
"""


import numpy as np
import numba as nb


def neg(a):
    return -a


def absv(a):  # 取绝对值
    return np.abs(a)


def intratsfftreal(a):  # 日内fft实数部分
    return np.fft.fft(a, axis=1).real / a.shape[1]  # 归一化


def intratsfftimag(a):  # 日内fft虚数部分
    return np.fft.fft(a, axis=1).imag / a.shape[1]  # 归一化
