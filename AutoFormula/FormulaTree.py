# Copyright (c) 2021 Dai HBG

"""
FormulaTree类定义了公式树变异的方法

开发日志：
2021-09-20
-- 更新：新增多个算子
2021-10-05
-- 更新：新增高频数据的解析
2021-10-22
-- 更新：Node类需要支持多个数字输入，数字优先级按照序号从低到高
-- 更新：新增多种运算符解析
-- 更新：修改运算逻辑，数字不单独新建一个子树，因为数字不可能还能挂载子树
-- 更新：FormulaTree需要判断并保证公式树最终产出的是一个二维的向量，于是Node节点需要记录该层节点的数据产出维度
-- 更新：公式树生成时可以指定生成维度，着决定了子树应该选择什么样的操作
-- 更新：更改公式树生成逻辑，例如data类型的节点一定不是数字
2021—11-01
-- 更新：新增傅里叶变换算子
-- 更新：多个循环算符使用向量化，提高速度
2021-11-09
-- 更新：新增对2_num_num_num类型算子的支持，包括parser和Node，以及树生成逻辑
2021-11-18
-- 更新：FormulaParser和Node移动到新的文件
2021-11-19
-- 更新：init方法中重复的生成子节点的操作成一个函数，便于维护
2021-11-26
-- 更新：可自定义挂载的data节点
2021-11-27
-- 更新：将算子字典单独存放
2021-11-29
-- 更新：可自定义item，即指定一个公式，该公式会被当成是一个data类型，在生成子树时会被随机选择挂载
-- 更新：可自定义树生成中dim_structure_probability
"""

import numpy as np
import sys

sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator/mytools/AutoFormula')
sys.path.append('C:/Users/Handsome Bad Guy/Desktop/Repositories/Low-Frequency-Spread-Estimator/mytools/AutoFormula')
from op_dic import *
from FormulaParser import Node, FormulaParser


class FormulaTree:
    def __init__(self, height=2, symmetric=False, data=None, intra_data=None, dim_operation_dic=None,
                 dim_structure_probability=None, daily_items=None, intra_items=None):
        """
        :param height: 树的最大深度
        :param symmetric: 是否需要对称
        :param data: 可自定义的日频数据
        :param intra_data: 可自定义日内数据
        :param dim_operation_dic: 可以自定义操作字典
        :param dim_structure_probability: 可以自定义2_2操作符到下一层为2_2还是3_2的抽样概率
        :param daily_items: 可以自定义生成一个2维矩阵的公式，这些公式将被当作是data随机抽样，且在节点处增加字段custom防止被修改
        :param intra_items: 可以自定义生成一个3维矩阵的公式，这些公式将被当作是intra_data随机抽样，且在节点处增加字段custom防止被修改
        """
        self.height = height
        self.symmetric = symmetric

        if data is None:
            data = ['open', 'high', 'low', 'close', 'tvr_ratio', 'volume', 'avg']
        self.data = data
        if intra_data is None:
            intra_data = ['intra_open', 'intra_high', 'intra_low', 'intra_close', 'intra_volume', 'intra_avg']
        self.intra_data = intra_data

        if dim_structure_probability is None:
            dim_structure_probability = [0.6, 0.4]
        self.dim_structure_probability = dim_structure_probability

        if daily_items is None:
            self.daily_items = []
        else:
            self.daily_items = daily_items  # list
        if intra_items is None:
            self.intra_items = []
        else:
            self.intra_items = intra_items  # list

        self.operation_dic = default_operation_dic  # 加载默认操作字典

        # 操作符输入的数据维度，0表示没有要求
        self.operation_input_dim_dic = {'csrank': 2, 'zscore': 2, 'neg': 0, 'csindneutral': 2, 'csind': 2, 'absv': 0,
                                        'wdirect': 2, 'tsrank': 2, 'tskurtosis': 2, 'tsskew': 2,
                                        'tsmean': 2, 'tsstd': 2, 'tsdelay': 0, 'tsdelta': 0, 'tsmax': 2,
                                        'tsmin': 2, 'tsmaxpos': 2, 'tsminpos': 2, 'powv': 0, 'tspct': 0,
                                        'intratsmax': 3, 'intratsmaxpos': 3, 'intratsmin': 3,
                                        'intratsminpos': 3, 'intratsmean': 3, 'intratsstd': 3,
                                        'add': 0, 'prod': 0, 'minus': 0, 'div': 0, 'lt': 0, 'le': 0, 'gt': 0, 'ge': 0,
                                        'tscorr': 2, 'tsregres': 2,
                                        'intratscorr': 3, 'intratsregres': 3,
                                        'tssubset': 2,
                                        'condition': 2, 'tsautocorr': 2,
                                        'intra_open': 3, 'intra_high': 3, 'intra_low': 3,
                                        'intra_close': 3, 'intra_avg': 3, 'intra_volume': 3,
                                        'intra_money': 3}

        # 操作符产出的维度需求，0表示可接受的维度不止一种，取决于操作。目前只有intra_data的维度输出不确定
        self.operation_output_dim_dic = {'csrank': 2, 'zscore': 2, 'neg': 0, 'csindneutral': 2, 'csind': 2, 'absv': 0,
                                         'wdirect': 2, 'tsrank': 2, 'tskurtosis': 2, 'tsskew': 2,
                                         'tsmean': 2, 'tsstd': 2, 'tsdelay': 0, 'tsdelta': 0, 'tsmax': 2,
                                         'tsmin': 2, 'tsmaxpos': 2, 'tsminpos': 2, 'powv': 0, 'tspct': 0,
                                         'intratsmax': 2, 'intratsmaxpos': 2, 'intratsmin': 2,
                                         'intratsminpos': 2, 'intratsmean': 2, 'intratsstd': 2,
                                         'add': 0, 'prod': 0, 'minus': 0, 'div': 0, 'lt': 0, 'le': 0, 'gt': 0, 'ge': 0,
                                         'tscorr': 2, 'tsregres': 2,
                                         'intratscorr': 2, 'intratsregres': 3,
                                         'tssubset': 2,
                                         'condition': 2, 'tsautocorr': 2,
                                         'intra_open': 0, 'intra_high': 0, 'intra_low': 0,
                                         'intra_close': 0, 'intra_avg': 0, 'intra_volume': 0,
                                         'intra_money': 0}

        # 从指定维度到指定维度的算子
        if dim_operation_dic is None:
            self.dim_operation_dic = default_dim_operation_dic  # 加载自定义维度操作符字典
        else:
            self.dim_operation_dic = dim_operation_dic

        # 得到一个算子到类型的字典
        dic = {}
        for key, value in self.operation_dic.items():
            for v in value:
                dic[v] = key
        self.operation_type_dic = dic

        self.parser = FormulaParser()

    @staticmethod
    def get_node_num(operation, operation_type, node):  # 用于根据operation生成其数字节点的数值
        """
        :param operation: 操作名称
        :param operation_type: 操作类型
        :param node: 被挂载节点
        :return:

        生成数字的逻辑是根据所需数字个数依次添加数字，期间检查某些特别的操作符的数字是否合法
        """
        if operation_type not in ['1_num', '1_num_num', '1_num_num_num', '2_num', '2_num_num', '2_num_num_num']:
            return

        # 所有需要数字的操作符都需要生成第一个数字
        # 第一个数字参数是delay的操作符，但应该靠近1
        if operation in ['tsdelta', 'tsdelay', 'tspct', ]:
            node.num_1 = \
                np.random.choice([i for i in range(1, 11)], 1, p=[(15 - i) / 95 for i in range(1, 11)])[
                    0]
        # 第一个数字参数是delay，但应该稍微长一些的操作符
        elif operation in ['tsquantile', 'intratsquantile', 'tsquantileupmean',
                           'tsquantiledownmean', 'intratsquantileupmean', 'intratsquantiledownmean',
                           'bitsquantile', 'biintratsquantile', 'bitsquantileupmean',
                           'bitsquantiledownmean', 'biintratsquantileupmean', 'biintratsquantiledownmean'
                           ]:
            p = np.array([35 - i for i in range(4, 30)]).astype(float)
            p /= np.sum(p)
            node.num_1 = \
                np.random.choice([i for i in range(4, 30)], 1, p=p)[0]  # 此时应该产生比较大的数
        else:
            node.num_1 = \
                np.random.choice([i for i in range(2, 32)], 1,
                                 p=[(32 - i) / 465 for i in range(2, 32)])[0]

        # 日内滤波器算子第一个算子是截断位置，应当去掉平凡的情况
        if operation in ['intratshpf', 'intratslpf']:  # 此时需要产生2到22之间的数
            node.num_1 = np.random.choice([i for i in range(2, 22)], 1)[0]
        if operation == 'powv':
            node.num_1 /= 10

        # 对于需要第二第三个数字的操作符，按需添加合法的第二第三数字
        if operation_type in ['1_num_num', '2_num_num', '1_num_num_num', '2_num_num_num']:
            node.num_2 = \
                np.random.choice([i for i in range(2, 32)], 1,
                                 p=[(32 - i) / 465 for i in range(2, 32)])[0]

            # 日内操作符的两个参数是start和end，需要单独生成
            if ('intra' in operation) and ('quantile' not in operation):
                proba = np.array([24 - i for i in range(23)])
                start = np.random.choice([i for i in range(23)], 1, p=proba / np.sum(proba))[0]
                proba = np.array(i for i in range(start + 1, 24))
                end = np.random.choice([i for i in range(start + 1, 24)], 1, p=proba / np.sum(proba))[0]
                node.num_1 = start
                node.num_2 = end

            if operation in ['tsfftreal', 'tsfftimag', 'tshpf', 'tslpf']:
                node.num_1 = np.random.choice([i for i in range(2, 30)], 1)[0]
                node.num_2 = np.random.choice([i for i in range(1, node.num_1)], 1)[0]

            # 时序分位数类操作符第二个数字需要挂载分位数
            if operation in ['tsquantile', 'intratsquantile', 'tsquantileupmean',
                             'tsquantiledownmean', 'intratsquantileupmean', 'intratsquantiledownmean',
                             'bitsquantile', 'biintratsquantile', 'bitsquantileupmean',
                             'bitsquantiledownmean', 'biintratsquantileupmean', 'biintratsquantiledownmean'
                             ]:
                node.num_2 = np.random.choice([i for i in range(1, 9)]) / 8

            # 日内分位数算子，需要重新生成所有数字
            if operation in ['intraquantile', 'intraquantileupmean', 'intraquantiledownmean'
                                                                     'biintratsquantile', 'biintraquantileupmean',
                             'biintraquantiledownmean']:
                proba = np.array([24 - i for i in range(23)])
                start = np.random.choice([i for i in range(23)], 1, p=proba / np.sum(proba))[0]
                proba = np.array(i for i in range(start + 1, 24))
                end = np.random.choice([i for i in range(start + 1, 24)], 1, p=proba / np.sum(proba))[0]
                node.num_1 = start
                node.num_2 = end
                node.num_3 = np.random.choice([i for i in range(1, 9)]) / 8  # 第三个节点挂载分位数

            if operation == 'tssubset':
                node.num_1 = np.random.choice([i for i in range(2, 30)], 1)[0]
                node.num_2 = np.random.choice([i for i in range(0, node.num_1 - 1)], 1)[0]
                node.num_3 = np.random.choice([i for i in range(node.num_2, node.num_1)], 1)[0]

    def get_node_data(self, variable_type, dim_structure=None):  # 生成操作符的data节点
        """
        :param variable_type: data或者intra_data
        :param dim_structure: 如果是intra_data，需要指定
        :return: Node
        """
        if variable_type == 'data':

            # 生成节点
            if len(self.daily_items) >= 1:
                if np.random.uniform() <= len(self.data) / (len(self.data) + len(self.daily_items)):
                    data = np.random.choice(self.data, 1)[0]  # 从指定的日频数据中选取
                    custom = False
                else:
                    data = self.parser.parse(np.random.choice(self.daily_items))
                    custom = True
                    data.custom = custom  # 该节点是自定义节点
            else:
                data = np.random.choice(self.data, 1)[0]  # 从指定的日频数据中选取
                custom = False

            # 生成随机delay
            if np.random.uniform() < 0.3:
                delay = np.random.choice([1, 2, 3, 4, 5], 1)[0]
                node_1 = Node(name='tsdelay', variable_type='operation', operation_type='1_num', num_1=delay,
                              dim_structure='2_2')  # tsdelay的dim_structure一定是2_2，因为数据类型是data
                if not custom:
                    node_1.left = Node(name=data, variable_type='data')
                else:
                    node_1.left = data
                return node_1
            else:
                if not custom:
                    return Node(name=data, variable_type='data')
                else:
                    return data

        # 日内数据
        elif variable_type == 'intra_data':
            data = np.random.choice(self.intra_data, 1)[0]  # 从指定的日内数据中选取
            proba = np.array([(i - 11.5) ** 2 for i in range(24)])
            num = np.random.choice([i for i in range(24)], 1, p=proba / np.sum(proba))[0]
            if (len(self.intra_items) >= 1) and (dim_structure != '2_2'):
                if np.random.uniform() > len(self.intra_data) / (len(self.intra_data) + len(self.intra_items)):
                    data = self.parser.parse(np.random.choice(self.intra_items))
                    custom = True
                    data.custom = custom  # 该节点是自定义节点
                else:
                    custom = False
            else:
                custom = False

            if np.random.uniform() < 0.7:
                if dim_structure in ['3_2', '3_3']:  # 此时一定不需要数字参数，而是由操作符操作成指定维度
                    if not custom:
                        return Node(name=data, variable_type='intra_data')
                    else:
                        return data
                else:
                    return Node(name=data, variable_type='intra_data', num_1=num)
            else:  # 此时生成一个随机delay的节点
                delay = np.random.choice([1, 2, 3, 4, 5], 1)[0]
                if dim_structure in ['3_2', '3_3']:
                    node_1 = Node(name='tsdelay', variable_type='operation', operation_type='1_num', num_1=delay,
                                  dim_structure=dim_structure)
                    if not custom:
                        node_1.left = Node(name=data, variable_type='intra_data')
                    else:
                        node_1.left = data
                else:
                    node_1 = Node(name='tsdelay', variable_type='operation', operation_type='1_num', num_1=delay,
                                  dim_structure='2_2')
                    node_1.left = Node(name=data, variable_type='intra_data', num_1=num)
                return node_1

    def init_tree(self, height, symmetric=False, dim_structure='2_2'):
        """
        :param height: 树的高度
        :param symmetric: 是否需要对称的树，默认非对称，这样生成的树多样性更好
        :param dim_structure: 维度结构，3_2表示该层节点接受3维矩阵，向上层返回2维矩阵
        :return: 返回一个公式树
        """
        """
        公式树生成的逻辑是：高度是1则退出递归，根据output的维度确定操作符
        生成逻辑应该改成首先随机采样下一层的数据维度，再采样操作符
        """

        # node.height = height  # 需要记录该节点代表的树的深度，以便之后的树的变异方法的使用
        if height == 1:  # 如果高度是1，直接生成叶子节点就可以返回
            # 首先随机采样一个操作符，注意要加上自定义的概率，以及对应挂载的数据类型
            if dim_structure == '2_2':  # 此时进入一个2维数据，应该允许日内数据
                operation = np.random.choice(self.dim_operation_dic['2_2'], 1)[0]
                data_dim = 2
                variable_type = np.random.choice(['data', 'intra_data'], 1, p=[0.5, 0.5])[0]
            elif dim_structure == '3_3':
                operation = np.random.choice(self.dim_operation_dic['3_3'], 1)[0]
                data_dim = 3
                variable_type = 'intra_data'
            else:
                operation = np.random.choice(self.dim_operation_dic['3_2'], 1)[0]
                data_dim = 2
                variable_type = 'intra_data'
            operation_type = self.operation_type_dic[operation]
            # input_dim = self.operation_input_dim_dic[operation]  # 操作符需要的维度
            node = Node(name=operation, variable_type='operation', operation_type=operation_type,
                        dim_structure=dim_structure)
            node.data_dim = data_dim
            # 接下来根据操作符随机采样挂载的数据
            if variable_type == 'data':
                node.left = self.get_node_data(variable_type=variable_type)
            else:
                node.left = self.get_node_data(variable_type=variable_type, dim_structure=dim_structure)
            # 然后根据操作符类型构造子节点
            if operation_type in ['1_num', '2_num', '1_num_num', '2_num_num',
                                  '1_num_num_num', '2_num_num_num']:  # 新增挂载数字
                self.get_node_num(operation, operation_type, node)
            if operation_type in ['2', '2_num', '2_num_num', '2_num_num_num',
                                  '3']:  # 此时根据variable_type选择挂载的第二个数据点，挂载在右子树
                if variable_type == 'data':
                    node.right = self.get_node_data(variable_type=variable_type)
                else:
                    node.right = self.get_node_data(variable_type=variable_type)
            if operation_type in ['3']:  # 此时根据variable_type选择挂载的第三个数据点，挂载到中间子树
                if variable_type == 'data':
                    node.middle = self.get_node_data(variable_type=variable_type)
                else:
                    node.middle = self.get_node_data(variable_type=variable_type)
            return node

        else:  # 此时递归生成子节点
            # 确定该层节点的结构以及随机采样下一层节点的dim_structure，目前以较大概率从日频数据挖掘信息
            if dim_structure == '2_2':
                operation = np.random.choice(self.dim_operation_dic['2_2'], 1)[0]
                data_dim = 2
                next_dim_structure = np.random.choice(['2_2', '3_2'], 1, p=self.dim_structure_probability)
            elif dim_structure == '3_3':
                operation = np.random.choice(self.dim_operation_dic['3_3'], 1)[0]
                data_dim = 3
                next_dim_structure = '3_3'
            else:
                operation = np.random.choice(self.dim_operation_dic['3_2'], 1)[0]
                data_dim = 2
                next_dim_structure = '3_3'
            operation_type = self.operation_type_dic[operation]
            node = Node(name=operation, variable_type='operation', operation_type=operation_type,
                        dim_structure=dim_structure)
            node.data_dim = data_dim

            if symmetric:  # 否则如果是对称，就根据操作类型递归生成子节点
                node.left = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                if operation_type in ['1_num', '2_num', '1_num_num', '2_num_num']:  # 此时挂载数字节点
                    if operation in ['tsdelta', 'tsdelay', 'tspct']:
                        node.num_1 = np.random.choice([i for i in range(1, 11)], 1,
                                                      p=[(15 - i) / 95 for i in range(1, 11)])[0]
                    else:
                        node.num_1 = np.random.choice([i for i in range(2, 32)], 1,
                                                      p=[(32 - i) / 465 for i in range(2, 32)])[0]
                    if operation_type in ['1_num_num', '2_num_num']:  # 此时就还要加一个num_2
                        node.num_2 = \
                            np.random.choice([i for i in range(2, 32)], 1,
                                             p=[(32 - i) / 465 for i in range(2, 32)])[0]
                if operation_type in ['2', '2_num', '2_num_num', '3']:  # 递归生成第二个节点，连接在右子树
                    node.right = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                if operation_type in ['3']:  # 递归生成第三个节点
                    node.middle = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                    return node
            else:  # 如果不对称，且运算符是双目的，则需要随机选取一边满足高度的约束
                if operation_type in ['1', '1_num', '1_num_num', '1_num_num_num']:  # 单目运算符下一层必然是高度-1
                    node.left = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                else:
                    if operation_type == '3':
                        left_or_right = np.random.choice([0, 1, 2], 1, p=[0.2, 0.4, 0.4])[0]  # 决定层数限制的节点
                        # 注意三目运算符中左中右是按顺序的，但是这里2代表middle
                    else:
                        left_or_right = np.random.choice([0, 1], 1)[0]
                    if left_or_right == 0:
                        node.left = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                        right_height = np.random.choice([i for i in range(1, height)], 1)[0]
                        node.right = self.init_tree(right_height, symmetric=symmetric, dim_structure=next_dim_structure)
                    elif left_or_right == 1:
                        node.right = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                        left_height = np.random.choice([i for i in range(1, height)], 1)[0]
                        node.left = self.init_tree(left_height, symmetric=symmetric, dim_structure=next_dim_structure)
                    else:  # 此时是3目运算符
                        node.middle = self.init_tree(height - 1, symmetric=symmetric, dim_structure=next_dim_structure)
                        left_height = np.random.choice([i for i in range(1, height)], 1)[0]
                        node.left = self.init_tree(left_height, symmetric=symmetric, dim_structure=next_dim_structure)
                        right_height = np.random.choice([i for i in range(1, height)], 1)[0]
                        node.right = self.init_tree(right_height, symmetric=symmetric,
                                                    dim_structure=next_dim_structure)

                self.get_node_num(operation, operation_type, node)  # 在函数内判断是否需要添加数字
                return node

    def change_data(self, tree, p=0.7):
        """
        :param tree: 需要被改变叶子节点数据的树
        :param p: 每个数据单独被改变的概率
        :return: 没有返回值，直接修改
        """
        if tree.custom:  # 自定义子树不得更改
            return
        if tree.variable_type == 'data':  # 此时数据节点一定没有数字
            if np.random.uniform() < p:
                variable_type = np.random.choice(['data', 'intra_data'], 1, p=[0.4, 0.6])[0]
                if variable_type == 'data':
                    tree.name = np.random.choice(self.data, 1)[0]
                else:
                    name = np.random.choice(self.intra_data, 1)[0]
                    proba = [(i - 11.5) ** 2 for i in range(24)]
                    num = np.random.choice([i for i in range(24)], 1, p=proba / np.sum(proba))[0]
                    tree.name = name
                    tree.num_1 = num

        elif tree.variable_type == 'intra_data':  # 日内数据直接随机替换
            if np.random.uniform() < p:
                variable_type = np.random.choice(['data', 'intra_data'], 1, p=[0.4, 0.6])[0]
                if tree.num_1 is None:
                    variable_type = 'intra_data'
                if variable_type == 'data':
                    tree.name = np.random.choice(self.data, 1)[0]
                    tree.num_1 = None
                else:
                    name = np.random.choice(self.intra_data, 1)[0]
                    proba = [(i - 11.5) ** 2 for i in range(24)]
                    num = np.random.choice([i for i in range(24)], 1, p=proba / np.sum(proba))[0]
                    tree.name = name
                    if tree.num_1 is not None:  # 只有存在num_1时才能替换
                        tree.num_1 = num
        else:
            if tree.num_1 is not None:  # 此时一定是操作符的第一个数字
                if np.random.uniform() < p:
                    self.get_node_num(operation=tree.name, operation_type=tree.operation_type, node=tree)
            if tree.left is not None:
                self.change_data(tree.left, p=p)
            if tree.right is not None:
                self.change_data(tree.right, p=p)
            if tree.middle is not None:
                self.change_data(tree.middle, p=p)

    def change_structure(self, a):  # 改变树的结构，可以选择是否局部更改
        pass

    def get_node_dim(self, tree):  # 递归计算树节点的维度
        if tree.variable_type == 'data':
            tree.data_dim = 2
        elif tree.variable_type == 'intra_data':
            if tree.num_1 is None:
                tree.data_dim = 2
            else:
                tree.data_dim = 3
        else:
            tree.data_dim = self.operation_output_dim_dic[tree.name]
            if tree.left is not None:
                self.get_node_dim(tree.left)
            if tree.middle is not None:
                self.get_node_dim(tree.middle)
            if tree.right is not None:
                self.get_node_dim(tree.right)

    def check_dim(self, tree):  # 递归检查公式树的节点运算维度是否正确，此时必须保证树的节点已经都有维度属性
        if tree.variable_type in ['data', 'intra_data']:
            return True
        if (tree.left is not None) and (not self.check_dim(tree.left)):
            return False
        if (tree.middle is not None) and (not self.check_dim(tree.middle)):
            return False
        if (tree.right is not None) and (not self.check_dim(tree.right)):
            return False
        if tree.operation_type in ['1', '1_num', '1_num_num']:
            return self.operation_input_dim_dic[tree.name] == tree.left.data_dim
        if tree.operation_type in ['2', '2_num', '2_num_num']:
            return (self.operation_input_dim_dic[tree.name] == tree.left.data_dim) and \
                   (self.operation_input_dim_dic[tree.name] == tree.right.data_dim)
        if tree.operation_type in ['3']:
            return (self.operation_input_dim_dic[tree.name] == tree.left.data_dim) and \
                   (self.operation_input_dim_dic[tree.name] == tree.right.data_dim) and \
                   (self.operation_input_dim_dic[tree.name] == tree.middle.data_dim)
