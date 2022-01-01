import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st
import multiprocessing
import os
import random
from tqdm import tqdm


class DataProcessor:
    def __init__(self, path_u=None, path_d=None, path_s=None):
        if path_u is None:
            path_u = r"E:/10min003_dataset/10min003u/"
        self.path_u = path_u
        if path_d is None:
            path_d = r"E:/10min003_dataset/10min003d/"
        self.path_d = path_d
        if path_s is None:
            path_s = r"E:/10min003s_dataset/10min003s/"
        self.path_s = path_s

    @staticmethod
    def get_data_tick(path):  # 得到所有文件。应该将度的文件保存在类里。
        f = os.listdir(path)
        data_tick = []
        stockinfo = []
        for info_0 in tqdm(f):
            stockinfo.append(info_0)
            domain = os.path.abspath(path)
            info = os.path.join(domain, info_0)
            data = pd.read_csv(info, na_values=['--  '], index_col=0, error_bad_lines=False, engine='python')
            data_tick.append(data)
        return data_tick

    @staticmethod
    def tick_3s_multi(data_tick, y):  # 基于n个文件200*20的矩阵，在矩阵上进行信息扩充，返回n个200*p的矩阵（其中p>=20）
        """
        :param data_tick: 包含所有DataFrame的列表
        :param y: 标签
        :return: x，y
        """
        data_tick_x = []
        for i in tqdm(range(len(data_tick))):
            data_1 = data_tick[i]
            data_1['bidprice1logreturn'] = data_1['bidprice1'].apply(lambda x: np.log(x)) - data_1['bidprice1']. \
                apply(lambda x: np.log(x)).shift()
            data_1['askprice1logreturn'] = data_1['askprice1'].apply(lambda x: np.log(x)) - data_1['askprice1']. \
                apply(lambda x: np.log(x)).shift()
            data_1['relativepricediff'] = (data_1['askprice1'] - data_1['bidprice1']) * 2 / \
                                          (data_1['askprice1'] + data_1['bidprice1'])
            data_1['bidvolumelogdiff'] = data_1['bidvolume1'].apply(lambda x: np.log(x)) - data_1['bidvolume1']. \
                apply(lambda x: np.log(x)).shift()
            data_1['askvolumelogdiff'] = data_1['askvolume1'].apply(lambda x: np.log(x)) - data_1['askvolume1']. \
                apply(lambda x: np.log(x)).shift()
            data_1['depth'] = (data_1['askvolume1'] + data_1['bidvolume1']) / 2
            data_1['slope'] = (data_1['askprice1'] - data_1['bidprice1']) / data_1['depth']
            data_1['qtyimbalance1'] = (data_1['askvolume1'] - data_1['bidvolume1']) / (data_1['askvolume1'] + data_1['bidvolume1']).apply(lambda x: np.sqrt(x))
            data_1['qtyimbalance2'] = (data_1['askvolume2'] - data_1['bidvolume2']) / (data_1['askvolume2'] + data_1['bidvolume2']).apply(lambda x: np.sqrt(x))
            data_1['qtyimbalance3'] = (data_1['askvolume3'] - data_1['bidvolume3']) / (data_1['askvolume3'] + data_1['bidvolume3']).apply(lambda x: np.sqrt(x))
            data_1['qtyimbalance4'] = (data_1['askvolume4'] - data_1['bidvolume4']) / (data_1['askvolume4'] + data_1['bidvolume4']).apply(lambda x: np.sqrt(x))
            data_1['qtyimbalance5'] = (data_1['askvolume5'] - data_1['bidvolume5']) / (data_1['askvolume5'] + data_1['bidvolume5']).apply(lambda x: np.sqrt(x))
            
            data_1['moneyimbalance1'] = (data_1['askvolume1']*data_1['askprice1'] - data_1['bidvolume1']*data_1['bidprice1']) /  \
                (data_1['askvolume1']*data_1['askprice1'] + data_1['bidvolume1']*data_1['bidprice1'])
            data_1['moneyimbalance2'] = (data_1['askvolume2']*data_1['askprice2'] - data_1['bidvolume2']*data_1['bidprice2']) /  \
                (data_1['askvolume2']*data_1['askprice2'] + data_1['bidvolume2']*data_1['bidprice2'])
            data_1['moneyimbalance3'] = (data_1['askvolume3']*data_1['askprice3'] - data_1['bidvolume3']*data_1['bidprice3']) /  \
                (data_1['askvolume3']*data_1['askprice3'] + data_1['bidvolume3']*data_1['bidprice3'])
            data_1['moneyimbalance4'] = (data_1['askvolume4']*data_1['askprice4'] - data_1['bidvolume4']*data_1['bidprice4']) /  \
                (data_1['askvolume4']*data_1['askprice4'] + data_1['bidvolume4']*data_1['bidprice4'])
            data_1['moneyimbalance5'] = (data_1['askvolume5']*data_1['askprice5'] - data_1['bidvolume5']*data_1['bidprice5']) /  \
                (data_1['askvolume5']*data_1['askprice5'] + data_1['bidvolume5']*data_1['bidprice5'])

            data_1.fillna(0)
            data_tick_x.append(data_1)
        data_tick_y = len(data_tick_x) * [y]
        return data_tick_x, np.array(data_tick_y)
    
    @staticmethod
    def tick_3s_single(data_tick):
        ##输入：data_tick是tick_3s_multi中返回的很多个dataframe
        ##输出：n*p的array,n是行数、样本量,p是列数、特征个数
        data_single = pd.DataFrame()
        for i in tqdm(range(len(data_tick))):
            data_1 = data_tick[i].copy()
            temp_list = []
            temp = (data_1['bidvolume1'] - data_1['askvolume1']) / (data_1['bidvolume1'] + data_1['askvolume1'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidvolume2'] - data_1['askvolume2']) / (data_1['bidvolume2'] + data_1['askvolume2'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidvolume3'] - data_1['askvolume3']) / (data_1['bidvolume3'] + data_1['askvolume3'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidvolume4'] - data_1['askvolume4']) / (data_1['bidvolume4'] + data_1['askvolume4'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidvolume5'] - data_1['askvolume5']) / (data_1['bidvolume5'] + data_1['askvolume5'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidvolume1'] * data_1['askprice1'] + data_1['askvolume1'] * data_1['bidprice1']) / (data_1['bidvolume1'] + data_1['askvolume1'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['askvolume1'] * data_1['askprice1'] + data_1['bidvolume1'] * data_1['bidprice1']) / (
                        data_1['bidvolume1'] + data_1['askvolume1'])
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidprice1'] + data_1['askprice1']) / 2
            temp_list.append(np.mean(temp.values))
            temp = (data_1['bidprice1'] + data_1['askprice1']) / 2
            temp_list.append(np.std(temp.values))
            data_1['mid-price'] = (data_1['bidprice1'] + data_1['askprice1']) / 2
            temp = (data_1['mid-price'] - data_1['mid-price'].shift()).apply(lambda x: 1* (x > 0))
            temp_list.append(np.nansum(temp.values)/199)
            temp = (data_1['mid-price'] - data_1['mid-price'].shift()).apply(lambda x: 1 * (x < 0))
            temp_list.append(np.nansum(temp.values)/199)
            temp = (data_1['mid-price'] - data_1['mid-price'].shift()).apply(lambda x: 1 * (x == 0))
            temp_list.append(np.nansum(temp.values)/199)
            del data_1['mid-price']

            data_single[str(i)] = temp_list
        data_single = np.array(data_single.T)
        return data_single

    @staticmethod
    def tick_3s_multi_standardized(data_tick, y):
        pass
        # tick_3s_all处理后的数据进行标准化的结果

    @staticmethod
    def getting_feature_array_x(df_multi,df_single,column_num_multi,column_num_single):
        ##输入：df_multi为tick_3s_multi中返回的很多个dataframe, colume_num_multi为每个dataframe中我们想出作为特征的列
        ##输入：df_single为tick_3s_single中返回的dataframe, colume_num_single为想要的具体特征
        ##输出：np.array形式,把两种信息整合在一起，每个样本为200*len(column_num_multi)+len(column_num_single)维的特征

        n = len(df_multi)
        data_tick_x = []
        for i in range(n):
            temp = []
            data1 = df_multi[i]
            for j in range(len(column_num_multi)):
                temp.extend(data1[data1.columns[column_num_multi[j]]].values)
            temp.extend(np.array(df_single[i][column_num_single]))

            data_tick_x.append(temp)
        return np.array(data_tick_x)
            

    def data_3s_feature_all(self):
        # path1是u类文件，path2是d类文件，path3是s类文件
        # function1选择进行数据处理的函数名（比如tick_3s_all_stardardized）
        # 输出中data_tick_u_x是n(样本量个数)个dataframe表格
        path1 = self.path_u
        path2 = self.path_d
        path3 = self.path_s
        function1 = self.tick_3s_multi

        data_tick_u = self.get_data_tick(path1)
        data_tick_d = self.get_data_tick(path2)
        data_tick_s = self.get_data_tick(path3)
        data_tick_u_x, data_tick_u_y = function1(data_tick_u, 1)
        data_tick_d_x, data_tick_d_y = function1(data_tick_d, -1)
        data_tick_s_x, data_tick_s_y = function1(data_tick_s, 0)
        return data_tick_u_x, data_tick_u_y, data_tick_d_x, data_tick_d_y, data_tick_s_x, data_tick_s_y
    
    
    def data_feature_input_part(self,r,column_num_multi,column_num_single,data_tick_u_x, data_tick_d_x, data_tick_s_x):
        # r为训练集所占数据比例
        data_tick_u_x_single = self.tick_3s_single(data_tick_u_x)
        data_tick_d_x_single = self.tick_3s_single(data_tick_d_x)
        data_tick_s_x_single = self.tick_3s_single(data_tick_s_x)

        data_tick_u_x = self.getting_feature_array_x(data_tick_u_x,data_tick_u_x_single,column_num_multi,column_num_single)
        data_tick_d_x = self.getting_feature_array_x(data_tick_d_x,data_tick_d_x_single,column_num_multi,column_num_single)
        data_tick_s_x = self.getting_feature_array_x(data_tick_s_x,data_tick_s_x_single,column_num_multi,column_num_single)

        sample_num_u = int(r * len(data_tick_u_x))
        select_u_list = [i for i in range(len(data_tick_u_x))]
        select_u = random.sample(select_u_list, sample_num_u)
        rest_u = list(set(select_u_list).difference(set(select_u)))
        sample_num_d = int(r * len(data_tick_d_x))
        select_d_list = [i for i in range(len(data_tick_d_x))]
        select_d = random.sample(select_d_list, sample_num_d)
        rest_d = list(set(select_d_list).difference(set(select_d)))
        sample_num_s = int(r * len(data_tick_s_x))
        select_s_list = [i for i in range(len(data_tick_s_x))]
        select_s = random.sample(select_s_list, sample_num_s)
        rest_s = list(set(select_s_list).difference(set(select_s)))

        # 统一用二维array处理，减少内存占用
        x_train = np.vstack([data_tick_u_x[select_u, :], data_tick_s_x[select_s, :], data_tick_d_x[select_d, :]])
        data_tick_u_y_train = np.ones(len(select_u))
        data_tick_d_y_train = -np.ones(len(select_d))
        data_tick_s_y_train = np.zeros(len(select_s))
        y_train = np.hstack([data_tick_u_y_train, data_tick_s_y_train, data_tick_d_y_train])

        x_test = np.vstack([data_tick_u_x[rest_u, :], data_tick_s_x[rest_s, :], data_tick_d_x[rest_d, :]])
        data_tick_u_y_test = np.ones(len(rest_u))
        data_tick_d_y_test = -np.ones(len(rest_d))
        data_tick_s_y_test = np.zeros(len(rest_s))
        y_test = np.hstack([data_tick_u_y_test, data_tick_s_y_test, data_tick_d_y_test])

        return x_train, y_train, x_test, y_test


