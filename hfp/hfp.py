# Copyright 2021-2022 Dai HBG


"""
该模块定义快速测试特征的总类

日志
2022-01-01
- init
"""


import numpy as np
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/High-Frequency-Predictor')
from DataLoader.DataLoader import DataLoader
from AutoFormula.AutoTester import AutoTester, Stats
from AutoFormula.AutoFormula import AutoFormula


class HFP:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/HFData',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData'):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path
        dl = DataLoader(data_path=data_path, back_test_data_path=back_test_data_path)
        self.datas = dl.load()
        self.tester = AutoTester()
        self.auto_formula = {key: AutoFormula(value) for key, value in self.datas.items()}

    def test_factor(self, formula: str, verbose: bool = True, start: int = 100,
                    end: int = 4600) -> dict:
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param verbose: 是否打印结果
        :param: start: 每日测试开始的snap
        :param: end: 每日测试结束的snap
        :return: 返回统计值以及该因子产生的信号矩阵的字典
        """
        to_return = {}
        for key, value in self.datas.items():
            stats, signal = self.auto_formula[key].test_formula(formula, value, start=start, end=end)
            to_return[key] = (stats, signal)
            if verbose:
                print('{} mean corr: {:.4f}, positive_corr_ratio: {:.4f}, corr_IR: {:.4f}'.
                      format(key, stats.mean_corr, stats.positive_corr_ratio, stats.corr_IR))
        return to_return
