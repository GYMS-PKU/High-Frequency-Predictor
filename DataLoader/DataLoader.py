# Copyright (c) 2021 Dai HBG


"""
该模块定义读取高频数据的方法

日志
2021-12-31
- init
"""


import numpy as np
import pandas as pd
import os


class Data:
    def __init__(self, code_order_dic: dict, order_code_dic: dict, date_position_dic: dict, position_date_dic: dict,
                 data_dic: dict, spread: np.array, spread_dic: dict = None, top: np.array = None):
        """
        :param code_order_dic: 股票代码到矩阵位置的字典
        :param order_code_dic: 矩阵位置到股票代码的字典
        :param date_position_dic: 日期到矩阵下标的字典
        :param data_dic: 所有的数据，形状一致
        :param spread: 默认使用的价差
        :param spread_dic: 可选的spread字典
        :param top: 初始的top
        """
        self.code_order_dic = code_order_dic
        self.order_code_dic = order_code_dic
        self.date_position_dic = date_position_dic
        self.position_date_dic = position_date_dic
        self.data_dic = data_dic
        self.spread = spread
        self.spread_dic = spread_dic
        self.top = top

    def get_real_date(self, start_date: str, end_date: str) -> (int, int):  # 用于获取起始日期对应的真正的数据起始位置
        """
        :param start_date: 任意输入的开始日期
        :param end_date: 任意输入的结束日期
        :return: 返回有交易的真正的起始日期对应的下标
        """
        tmp_start = start_date.split('-')
        i = 0
        while True:
            s = datetime.date(int(tmp_start[0]), int(tmp_start[1]), int(tmp_start[2])) + datetime.timedelta(days=i)
            try:
                start = self.date_position_dic[s]
                break
            except KeyError:
                i += 1
        i = 0
        tmp_end = end_date.split('-')
        while True:
            s = datetime.date(int(tmp_end[0]), int(tmp_end[1]), int(tmp_end[2])) + datetime.timedelta(days=i)
            try:
                end = self.date_position_dic[s]
                break
            except KeyError:
                i -= 1
        return start, end


class DataLoader:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/DailyData/data',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData'):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path

    def load(self) -> Data:
        """
        :return: data，包含data_dic，分别为OHLC四个价格；spread，价差
        """
        days = os.listdir(self.data_path)
        spread = np.zeros((len(days), 2081))  # 默认2081只股票
        relative_spread = np.zeros((len(days), 2081))  # 默认2081只股票
        names = ['open', 'low', 'high', 'close']  # 字段
        code_order_dic = {}
        order_code_dic = {}
        date_position_dic = {}
        position_date_dic = {}
        data_dic = {name: np.zeros((len(days), 2081)) for name in names}

        codes = pd.read_csv('{}/{}'.format(self.data_path, days[0]))['Unnamed: 0'].values
        for i in range(len(codes)):
            stock_code = str(codes[i])
            if len(stock_code) < 6:
                stock_code = '0' * (6 - len(stock_code)) + stock_code
            else:
                stock_code = str(stock_code)
            order_code_dic[i] = stock_code
            code_order_dic[codes[i]] = i

        day_num = 0
        for day in days:
            date_position_dic[datetime.date(2020, int(day[:2]), int(day[2:4]))] = day_num
            position_date_dic[day_num] = datetime.date(2020, int(day[:2]), int(day[2:4]))
            df = pd.read_csv('{}/{}'.format(self.data_path, day))
            for name in names:
                data_dic[name][day_num] = df[name].values.copy()
            spread[day_num] = df['bid_ask_spread'].values.copy()
            relative_spread[day_num] = df['relative_spread'].values.copy()
            day_num += 1

        data = Data(code_order_dic=code_order_dic, order_code_dic=order_code_dic,
                    position_date_dic=position_date_dic, date_position_dic=date_position_dic,
                    spread=spread, spread_dic={'spread': spread, 'relative_spread': relative_spread},
                    data_dic=data_dic)
        return data
