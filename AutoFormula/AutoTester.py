# Copyright (c) 2021 Dai HBG

"""
AutoTester
该模块用于测试信号对于高频价差的线性预测力

日志
2021-12-19
- 初始化

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
    def test(signal: np.array, spread: np.array, top: np.array = None, method: str = 'cs',
             corr_type: str = 'linear') -> Stats:
        """
        :param signal: 信号矩阵
        :param spread 和信号矩阵形状一致的高频收益率矩阵
        :param top: 每个时间截面上进入截面的股票位置
        :param method: 计算方式
        :param corr_type: linear或者log
        :return:
        """
        signal[np.isnan(signal)] = 0
        if top is None:
            top = (signal != 0) & (~np.isnan(signal)) & (~np.isinf(signal)) & (~np.isnan(spread))

        if method == 'cs':  # 截面相关系数
            corr = np.zeros(spread.shape[0])
            for i in range(spread.shape[0]):
                se = ~np.isnan(spread[i])
                if corr_type == 'linear':
                    corr[i] = np.corrcoef(spread[i][se], signal[i][se])[0, 1]
                elif corr_type == 'log':
                    corr[i] = np.corrcoef(np.log(spread[i][se]), signal[i][se])[0, 1]
            mean_corr = np.nanmean(corr)
            corr_IR = mean_corr / np.nanstd(corr)
            positive_corr_ratio = np.sum(corr > 0) / np.sum(~np.isnan(corr))
            return Stats(corr=corr, mean_corr=mean_corr, corr_IR=corr_IR, positive_corr_ratio=positive_corr_ratio)
        elif method == 'ts':  # 时序相关系数
            corr = np.zeros(spread.shape[1])
            for i in range(spread.shape[1]):
                se = ~np.isnan(spread[:, i])
                if corr_type == 'linear':
                    corr[i] = np.corrcoef(spread[:, i][se], signal[:, i][se])[0, 1]
                else:
                    corr[i] = np.corrcoef(np.log(spread[:, i][se]), signal[:, i][se])[0, 1]
            mean_corr = np.nanmean(corr)
            corr_IR = mean_corr / np.nanstd(corr)
            positive_corr_ratio = np.sum(corr > 0) / np.sum(~np.isnan(corr))
            return Stats(corr=corr, mean_corr=mean_corr, corr_IR=corr_IR, positive_corr_ratio=positive_corr_ratio)
