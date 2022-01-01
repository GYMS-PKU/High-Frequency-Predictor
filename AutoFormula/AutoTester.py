# Copyright (c) 2022 Dai HBG

"""
AutoTester
该模块用于测试信号对于高频价差的线性预测力

日志
2022-01-01
- init
"""

import numpy as np


class Stats:
    def __init__(self, corr: np.array = None, mean_corr: float = 0, corr_IR: float = 0,
                 positive_corr_ratio: float = 0):
        """
        :param corr: 相关系数
        :param mean_corr: 平均相关系数
        :param corr_IR:
        :param positive_corr_ratio: 相关系数为正的比例
        """
        self.corr = corr
        self.mean_corr = mean_corr
        self.corr_IR = corr_IR
        self.positive_corr_ratio = positive_corr_ratio


class AutoTester:
    def __init__(self):
        pass

    @staticmethod
    def test(signal: np.array, ret: np.array, start: int = 100, end: int = 4600) -> Stats:
        """
        :param signal: 信号矩阵
        :param ret: 收益率矩阵
        :param start: 开始时间
        :param end: 结束时间
        :return:
        """
        signal[np.isnan(signal)] = 0
        ret[np.isnan(ret)] = 0

        corr = np.zeros(signal.shape[1])
        for i in range(signal.shape[1]):
            corr[i] = np.corrcoef(ret[:, i], signal[:, i])[0, 1]
        mean_corr = np.nanmean(corr)
        corr_IR = mean_corr / np.nanstd(corr)
        positive_corr_ratio = np.sum(corr > 0) / np.sum(~np.isnan(corr))
        return Stats(corr=corr, mean_corr=mean_corr, corr_IR=corr_IR, positive_corr_ratio=positive_corr_ratio)
