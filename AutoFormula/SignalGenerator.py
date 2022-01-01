# Copyright (c) 2021 Dai HBG

"""
该代码定义的SignalGenerator类将解析一个公式，然后递归地计算signal值
v1.0
默认所有操作将nan替换成0

开发日志：
2021-09-13
-- 新增：csindneutral算子，获得行业中性的信号
-- 更新：为了方便算子计算，SignalGenerator类需要传入一个data类进行初始化
2021-10-05
-- 新增：分钟数据算子
2021-11-01
-- 更新：循环算子使用向量化，提高速度
-- 新增：傅里叶变换算子
-- 更新：信号矩阵逻辑，没有值的地方以np.nan填充，而不是0
2021-11-18
-- 修复：多个和argsort相关的bug
2021-11-20
-- 更新：新增截面类算子
-- 更新：函数变量命名更改，含义清晰化
2021-11-22
-- 更新：算子分开定义，结构清晰化
2021-11-27
-- 更新：新增离散分段算子，希望提高信号鲁棒性
"""


import numba as nb
import numpy as np
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator'
                '/mytools/AutoFormula/operations')
sys.path.append('C:/Users/Handsome Bad Guy/Desktop/Repositories/Low-Frequency-Spread-Estimator'
                '/mytools/AutoFormula/operations')
from one import *
from one_num import *
from one_num_num import *
from one_num_num_num import *
from two import *
from two_num import *
from two_num_num import *
from two_num_num_num import *


class SignalGenerator:
    def __init__(self, data):
        """
        :param data: Data类的实例
        """
        self.operation_dic = {}
        self.get_operation()
        self.data = data

        # 单独注册需要用到额外信息的算子
        self.operation_dic['zscore'] = self.zscore
        self.operation_dic['csrank'] = self.csrank
        self.operation_dic['csindneutral'] = self.csindneutral
        self.operation_dic['csind'] = self.csind
        self.operation_dic['truncate'] = self.truncate
        self.operation_dic['marketbeta'] = self.marketbeta
        self.operation_dic['discrete'] = self.discrete

        """
        截面算子，因为要调用top
        """

    def csrank(self, a):
        b = a.copy()  # 保持a中的nan
        for i in range(len(a)):
            n = np.sum(~np.isnan(b[i, self.data.top[i]]))
            if n == 0:  # 全是nan就不排序了
                continue
            tmp = b[i, self.data.top[i]].copy()
            valid_tmp = tmp[~np.isnan(tmp)].copy()  # 取出不是nan的部分
            pos = valid_tmp.argsort()
            for j in range(len(pos)):
                valid_tmp[pos[j]] = j
            valid_tmp /= (len(valid_tmp) - 1)
            tmp[~np.isnan(tmp)] = valid_tmp
            b[i, self.data.top[i]] = tmp
        return b

    def zscore(self, a):
        s = a.copy()
        for i in range(len(a)):
            if np.sum(~np.isnan(s[i][self.data.top[i]])) <= 1:
                continue
            s[i][self.data.top[i]] -= np.nanmean(b[i][self.data.top[i]])
            b[i][self.data.top[i]] /= np.nanstd(b[i][self.data.top[i]])
            b[i][(self.data.top[i]) & (b[i] > 3)] = 3
            b[i][(self.data.top[i]) & (b[i] < -3)] = -3
        return b

    def csindneutral(self, a):  # 截面中性化，暂时先使用申万二级行业，之后需要加入可选行业中性化
        s = a.copy()
        ind = self.data.industry['sws']  # 申万二级行业的位置
        for i in range(len(s)):
            ind_num_dic = {}  # 存放行业总数
            ind_sum_dic = {}  # 存放行业总值
            for j in list(set(ind[i])):
                ind_num_dic[j] = np.sum(ind[i] == j)
                ind_sum_dic[j] = np.sum(a[i, ind[i] == j])
            for key in ind_sum_dic.keys():
                ind_sum_dic[key] /= ind_num_dic[key]
            for j in range(s.shape[1]):
                s[i, j] = a[i, j] - ind_sum_dic[ind[i, j]]  # 减去行业平均，如果是没有出现过的行业，那么就是0
        return s

    def csind(self, a):  # 截面替换成所处行业的均值
        s = a.copy()
        ind = self.data.industry['sws']  # 申万二级行业的位置
        for i in range(len(s)):
            ind_num_dic = {}  # 存放行业总数
            ind_sum_dic = {}  # 存放行业总值
            for j in list(set(ind[i])):
                ind_num_dic[j] = np.sum(ind[i] == j)
                ind_sum_dic[j] = np.sum(a[i, ind[i] == j])
            for key in ind_sum_dic.keys():
                ind_sum_dic[key] /= ind_num_dic[key]
            for j in range(s.shape[1]):
                s[i, j] = ind_sum_dic[ind[i, j]]  # 减去行业平均，如果是没有出现过的行业，那么就是0
        return s

    def truncate(self, a, s, e):  # 将过大过小的信号截断为平均值，注意是平均值
        """
        :param a: 数据
        :param s: 起始
        :param e: 结束
        :return: 截断后的信号
        """
        b = self.csrank(a)
        sig = a.copy()
        for i in range(len(a)):
            mean = np.mean(s[i, self.data.top[i]])
            sig[i, self.data.top[i] & ((b[self.data.top[i]] < s) or (b[self.data.top[i]] > e))] = mean
        return sig

    def marketbeta(self, a, ts_window):  # 获得信号和市场平均回望ts_window的beta系数
        s = np.zeros(a.shape)
        if ts_window < 2:
            ts_window = 2  # 至少回望两天
        mar_mean = np.zeros(len(a))
        for i in range(len(a)):
            if np.sum(~np.isnan(a[i, self.data.top[i]])) > 0:
                mar_mean[i] = np.nanmean(a[i, self.data.top[i]])
            else:
                mar_mean[i] = np.nan
        tmp_a = np.zeros(ts_window, a.shape[1], a.shape[0])  # 必须是ts_window * cs * ts
        tmp_a[0] = a.copy().T
        for i in range(1, num):
            tmp_a[i, :, i:] = a[:-i].T  # 第i列存放delay i天的数据
        tmp_m = np.zeros((num, a.shape[1], a.shape[0]))
        tmp_m[0] = mar_mean
        for i in range(1, num):
            tmp_m[i, :, i:] = mar_mean[:-i]  # 第i列存放delay i天的数据
        tmp_a = tmp_a.transpose(0, 2, 1)
        tmp_m = tmp_m.transpose(0, 2, 1)
        tmp_a -= np.nanmean(tmp_a, axis=0)
        tmp_m -= np.nanmean(tmp_m, axis=0)
        s[num - 1:] = (np.nanmean(tmp_a * tmp_m,
                                  axis=0) / (np.nanstd(tmp_b, axis=0)))[num - 1:]
        s[:num - 1] = np.nan
        return s

    def discrete(self, a, num):  # 离散化算子，将截面信号离散成0到num-1的整数
        b = a.copy()  # 复制主要是保持a中本来是nan的部分也为nan
        for i in range(len(a)):
            n = np.sum(~np.isnan(b[i, self.data.top[i]]))
            if n == 0:  # 说明全是nan
                continue
            tmp = b[i, self.data.top[i]].copy()
            valid_tmp = tmp[~np.isnan(tmp)].copy()  # 取出不是nan的部分
            pos = valid_tmp.argsort()
            for j in range(num-1):
                se = (j * (len(pos) // num) <= pos) & (pos < (j + 1) * (len(pos) // num))
                valid_tmp[pos[se]] = j
            se = (num - 1) * (len(pos) // num) <= pos
            valid_tmp[pos[se]] = num - 1
            tmp[~np.isnan(tmp)] = valid_tmp  # 排序后再赋值回来
            b[i, self.data.top[i]] = tmp
        return b

    def get_operation(self):

        # 1型算符
        self.operation_dic['neg'] = neg
        self.operation_dic['absv'] = absv
        self.operation_dic['intratsfftreal'] = intratsfftreal
        self.operation_dic['intratsfftimag'] = intratsfftimag

        # 1_num型运算符
        self.operation_dic['powv'] = powv
        self.operation_dic['tsmax'] = tsmax
        self.operation_dic['intratsmax'] = intratsmax
        self.operation_dic['tsmaxpos'] = tsmaxpos
        self.operation_dic['tsmin'] = tsmin
        self.operation_dic['tsminpos'] = tsminpos
        self.operation_dic['tsdelay'] = tsdelay
        self.operation_dic['tsdelta'] = tsdelta
        self.operation_dic['tspct'] = tspct
        self.operation_dic['tsstd'] = tsstd
        self.operation_dic['tsmean'] = tsmean
        self.operation_dic['tskurtosis'] = tskurtosis
        self.operation_dic['tsskew'] = tsskew
        self.operation_dic['wdirect'] = wdirect
        self.operation_dic['tsrank'] = tsrank
        self.operation_dic['intratshpf'] = intratshpf
        self.operation_dic['intratslpf'] = intratslpf

        # 1_num_num型算子
        self.operation_dic['intratsmax'] = intratsmax
        self.operation_dic['intratsmaxpos'] = intratsmaxpos
        self.operation_dic['intratsmin'] = intratsmin
        self.operation_dic['intratsminpos'] = intratsminpos
        self.operation_dic['intratsstd'] = intratsstd
        self.operation_dic['intratsmean'] = intratsmean
        self.operation_dic['intratskurtosis'] = intratskurtosis
        self.operation_dic['intratskew'] = intratskew
        self.operation_dic['tsautocorr'] = tsautocorr
        self.operation_dic['tsfftreal'] = tsfftreal
        self.operation_dic['tsfftimag'] = tsfftimag
        self.operation_dic['tshpf'] = tshpf
        self.operation_dic['tslpf'] = tslpf
        self.operation_dic['tsquantile'] = tsquantile
        self.operation_dic['intratsquantile'] = intratsquantile
        self.operation_dic['tsquantileupmean'] = tsquantileupmean
        self.operation_dic['tsquantiledownmean'] = tsquantiledownmean
        self.operation_dic['intratsquantileupmean'] = intratsquantileupmean
        self.operation_dic['intratsquantiledownmean'] = intratsquantiledownmean

        # 1_num_num_num型算符
        self.operation_dic['intraquantile'] = intraquantile
        self.operation_dic['intraquantileupmean'] = intraquantileupmean
        self.operation_dic['intraquantiledownmean'] = intraquantiledownmean

        # 2型运算符
        self.operation_dic['add'] = add
        self.operation_dic['minus'] = minus
        self.operation_dic['prod'] = prod
        self.operation_dic['div'] = div
        self.operation_dic['intratsregres'] = intratsregres
        self.operation_dic['lt'] = lt
        self.operation_dic['le'] = le
        self.operation_dic['gt'] = gt
        self.operation_dic['ge'] = ge

        # 2_num型运算符
        self.operation_dic['tsregres'] = tsregres
        self.operation_dic['tscorr'] = tscorr
        self.operation_dic['tscov'] = tscov

        # 2_num_num型运算符
        self.operation_dic['bitsquantile'] = bitsquantile
        self.operation_dic['biintratsquantile'] = biintratsquantile
        self.operation_dic['bitsquantileupmean'] = bitsquantileupmean
        self.operation_dic['bitsquantiledownmean'] = bitsquantiledownmean
        self.operation_dic['biintratsquantileupmean'] = biintratsquantileupmean
        self.operation_dic['biintratsquantiledownmean'] = biintratsquantiledownmean

        def condition(a, b, c):
            """
            :param a: 条件，一个布尔型矩阵
            :param b: 真的取值
            :param c: 假的取值
            :return: 信号
            """
            s = np.zeros(a.shape)
            for i in range(len(a)):
                if type(b) == int or type(b) == float:
                    s[i, a[i]] = b
                else:
                    s[i, a[i]] = b[i, a[i]]
                if type(c) == int or type(c) == float:
                    s[i, ~a[i]] = c
                else:
                    s[i, ~a[i]] = c[i, ~a[i]]
            return s

        self.operation_dic['condition'] = condition

        # 2_num_num_num算符
        self.operation_dic['tssubset'] = tssubset
        self.operation_dic['biintraquantile'] = biintraquantile
        self.operation_dic['biintraquantileupmean'] = biintraquantileupmean
        self.operation_dic['biintraquantiledownmean'] = biintraquantiledownmean
