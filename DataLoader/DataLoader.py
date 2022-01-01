# Copyright 2021-2022 Dai HBG


"""
该模块定义读取高频数据的方法

日志
2021-12-31
- init
"""


import numpy as np
import pandas as pd
import os


class Data:  # 存放一只股票的所有交易日的日内高频数据
    def __init__(self, data_dic: dict, ret: np.array):
        """
        :param data_dic: 数据字典，是一个
        :param ret: 收益率
        """
        self.data_dic = data_dic
        self.ret = ret


class DataLoader:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/HFData',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData'):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path

    def load(self, stock_num: int = 1, stock_list: list = None) -> dict:
        """
        :param stock_num: 需要研究的股票数量
        :param stock_list: 需要研究的股票列表
        :return: dict，为所有股票的字典，值是Data
        """
        if stock_list is None:
            stock_list = os.listdir('{}/data'.format(self.data_path))
            stock_list = sorted(stock_list)
            stock_list = stock_list[:stock_num]
        datas = {}

        names = ['BidSize1', 'BidSize2', 'BidSize3', 'BidSize4', 'BidSize5',
                 'BidPX1', 'BidPX2', 'BidPX3', 'BidPX4', 'BidPX5',
                 'OfferSize1', 'OfferSize2', 'OfferSize3', 'OfferSize4', 'OfferSize5',
                 'OfferPX1', 'OfferPX2', 'OfferPX3', 'OfferPX4', 'OfferPX5',
                 'mid_price', 'OrderCount', 'TradeCount', 'ret']
        for stock in stock_list:
            stock_data = {name: [] for name in names}  # 最终得到一个snapshot_num * days的矩阵，逻辑是每日可以看成是独立样本
            ret = []
            years = os.listdir('{}/data/{}'.format(self.data_path, stock))
            for year in years:
                days = os.listdir('{}/data/{}/{}'.format(self.data_path, stock, year))
                for day in days:
                    snapshot = pd.read_csv('{}/data/{}/{}/{}/snapshot.csv'.format(self.data_path, stock, year, day))
                    ret.append(snapshot['ret'])  # 3s收益率
                    for name in names:
                        stock_data[name].append(snapshot[name].values)
            ret = np.vstack(ret).T
            for name in names:
                stock_data[name] = np.vstack(stock_data[name]).T
            datas[stock] = Data(data_dic=stock_data, ret=ret)
        return datas
