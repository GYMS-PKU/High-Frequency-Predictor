# Copyright (c) 2021 Dai HBG

"""
FormulaParser用于解析字符串公式，Node定义了公式树节点

开发日志
2021-11-18
-- 初始化
2021-11-25
-- 新增多个算子的支持
"""


import sys
sys.path.append('.')
from op_dic import *


class Node:
    def __init__(self, name, variable_type, operation_type=None,
                 left=None, right=None, middle=None, num_1=None, num_2=None, num_3=None, dim_structure=None):
        """
        :param name: 操作或者数据的名字
        :param variable_type: 变量类型，data指的是数据，operation指的是算符，intra_data指的是日内数据
        :param operation_type: 算符类型，数字指的是多少目运算符，num指的是需要传入一个确定数字而不是运算结果
        :param left: 左子树，可以挂载一颗公式树或者数据节点
        :param middle: 中间子树，可以挂载一颗公式树或者数据节点，也可以挂载数字
        :param right: 右子树，可以挂载一颗公式树或者数据节点
        :param num_1: 数字，如果算符需要传入一个数字，或者日内数据需要一个数字
        :param num_2: 数字，如果算符需要传入一个数字，或者日内数据需要一个数字
        :param num_3: 数字，如果算符需要传入一个数字，或者日内数据需要一个数字
        :param dim_structure: 该节点的维度结构，如果是操作符则有2_2，3_2，3_3三种，如果是数据节点则为None
        """
        self.name = name
        self.variable_type = variable_type
        if self.variable_type == 'operation':
            self.operation_type = operation_type
        self.left = left
        self.middle = middle
        self.right = right
        self.num_1 = num_1
        self.num_2 = num_2
        self.num_3 = num_3
        self.dim_structure = dim_structure
        self.data_dim = None  # 记录当前节点产出的数据维度
        self.custom = False  # 表示改节点开始的树是自定义公式，不得被更改


class FormulaParser:
    def __init__(self):
        self.operation_dic = default_operation_dic

        dic = {}
        for key, value in self.operation_dic.items():
            for v in value:
                dic[v] = key
        self.operation_type_dic = dic

    def parse(self, s):
        """
        :param s: 待解析字符串
        :return: 返回一棵树
        """
        if '{' not in s:  # 此时保证不是数字，因为只要是数字类型一定直接挂载到对应层的节点中
            if 'intra' not in s:
                return Node(name=s, variable_type='data')
            else:
                return Node(name=s, variable_type='intra_data')
        elif s[0] == '{':  # 此时对应的一定是日内数据，格式为{intra_data, int}
            # 定位中间的逗号
            a = 0
            while s[a] != ',':
                a += 1
            return Node(name=s[1:a], variable_type='intra_data', num_1=int(s[a + 1:-1]))
        else:
            # 定位到名称
            a = 0
            while s[a] != '{':
                a += 1
            name = s[:a]
            node = Node(name=name, variable_type='operation', operation_type=self.operation_type_dic[name])
            if self.operation_type_dic[name] == '1':
                node.left = self.parse(s[a + 1:-1])
                return node
            if self.operation_type_dic[name] == '1_num':
                # 定位中间的逗号
                b = len(s) - 2
                while s[b] != ',':
                    b -= 1
                if '.' in s[b + 1:len(s) - 1]:  # 需要判断是整数还是小数
                    num = float(s[b + 1:len(s) - 1])
                else:
                    num = int(s[b + 1:len(s) - 1])
                node.num_1 = num  # 数字直接挂载到这个节点上
                node.left = self.parse(s[a + 1:b])
                return node
            if self.operation_type_dic[name] == '1_num_num':
                b = len(s) - 2
                while s[b] != ',':
                    b -= 1
                if '.' in s[b + 1:len(s) - 1]:  # 需要判断是整数还是小数
                    num_2 = float(s[b + 1:len(s) - 1])
                else:
                    num_2 = int(s[b + 1:len(s) - 1])
                node.num_2 = num_2
                c = b - 1
                while s[c] != ',':
                    c -= 1
                if '.' in s[c + 1:b]:  # 需要判断是整数还是小数
                    num_1 = float(s[c + 1:b])
                else:
                    num_1 = int(s[c + 1:b])
                node.num_1 = num_1
                node.left = self.parse(s[a + 1:c])
                return node

            if self.operation_type_dic[name] == '1_num_num_num':
                # 定位数字
                b = len(s) - 2
                while s[b] != ',':
                    b -= 1
                if '.' in s[b + 1:len(s) - 1]:  # 需要判断是整数还是小数
                    num_3 = float(s[b + 1:len(s) - 1])
                else:
                    num_3 = int(s[b + 1:len(s) - 1])
                node.num_3 = num_3
                c = b - 1
                while s[c] != ',':
                    c -= 1
                if '.' in s[c + 1:b]:  # 需要判断是整数还是小数
                    num_2 = float(s[c + 1:b])
                else:
                    num_2 = int(s[c + 1:b])
                node.num_2 = num_2
                d = c - 1
                while s[d] != ',':
                    d -= 1
                if '.' in s[d + 1:c]:  # 需要判断是整数还是小数
                    num_1 = float(s[d + 1:c])
                else:
                    num_1 = int(s[d + 1:c])
                node.num_1 = num_1

                node.left = self.parse(s[a + 1:d])
                return node

            if self.operation_type_dic[name] == '2':
                # 定位中间的逗号
                left_num = 0
                right_num = 0
                b = len(s) - 2
                while True:
                    if s[b] == '}':
                        right_num += 1
                    if s[b] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b -= 1
                if left_num == 0:
                    b += 1
                c = b - 1  # 此时c的位置是算子最后一位符号
                while s[c] != ',':
                    c -= 1
                # 由于二元算子需要同时处理数字和data的情形，因此需要判断是不是数字
                try:
                    tmp = float(s[c + 1:len(s) - 1])
                    if (tmp - int(tmp)) < 0.01:  # 此时认为输入的是一个整数
                        node.num_1 = int(tmp)
                    else:
                        node.num_1 = tmp
                except ValueError:
                    node.right = self.parse(s[c + 1:len(s) - 1])
                try:
                    tmp = float(s[a + 1:c])
                    node.num_1 = tmp
                except ValueError:
                    node.left = self.parse(s[a + 1:c])
                return node
            if self.operation_type_dic[name] == '2_num':
                # 定位数字
                b = len(s) - 2
                while s[b] != ',':
                    b -= 1
                if '.' in s[b + 1:len(s) - 1]:  # 需要判断是整数还是小数
                    num_1 = float(s[b + 1:len(s) - 1])
                else:
                    num_1 = int(s[b + 1:len(s) - 1])
                node.num_1 = num_1
                # 定位第二个逗号
                c = b - 1
                right_num = 0
                left_num = 0
                while True:
                    if s[c] == '}':
                        right_num += 1
                    if s[c] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    c -= 1
                if left_num == 0:
                    c += 1
                d = c - 1  # 此时d的位置是算子最后一位符号
                while s[d] != ',':
                    d -= 1
                node.right = self.parse(s[d + 1:b])
                node.left = self.parse(s[a + 1:d])
                return node
            if self.operation_type_dic[name] == '2_num_num':
                # 定位数字
                b = len(s) - 2
                while s[b] != ',':
                    b -= 1
                if '.' in s[b + 1:len(s) - 1]:  # 需要判断是整数还是小数
                    num_2 = float(s[b + 1:len(s) - 1])
                else:
                    num_2 = int(s[b + 1:len(s) - 1])
                node.num_2 = num_2
                c = b - 1
                while s[c] != ',':
                    c -= 1
                if '.' in s[c + 1:b]:  # 需要判断是整数还是小数
                    num_1 = float(s[c + 1:b])
                else:
                    num_1 = int(s[c + 1:b])
                node.num_1 = num_1

                # 定位第二个逗号
                d = c - 1
                right_num = 0
                left_num = 0
                while True:
                    if s[d] == '}':
                        right_num += 1
                    if s[d] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    d -= 1
                if left_num == 0:
                    d += 1
                e = d - 1  # 此时d的位置是算子最后一位符号
                while s[e] != ',':
                    e -= 1
                node.right = self.parse(s[e + 1:c])
                node.left = self.parse(s[a + 1:e])
                return node

            if self.operation_type_dic[name] == '2_num_num_num':
                # 定位数字
                b = len(s) - 2
                while s[b] != ',':
                    b -= 1
                if '.' in s[b + 1:len(s) - 1]:  # 需要判断是整数还是小数
                    num_3 = float(s[b + 1:len(s) - 1])
                else:
                    num_3 = int(s[b + 1:len(s) - 1])
                node.num_3 = num_3
                c = b - 1
                while s[c] != ',':
                    c -= 1
                if '.' in s[c + 1:b]:  # 需要判断是整数还是小数
                    num_2 = float(s[c + 1:b])
                else:
                    num_2 = int(s[c + 1:b])
                node.num_2 = num_2
                d = c - 1
                while s[d] != ',':
                    d -= 1
                if '.' in s[d + 1:c]:  # 需要判断是整数还是小数
                    num_1 = float(s[d + 1:c])
                else:
                    num_1 = int(s[d + 1:c])
                node.num_1 = num_1

                # 定位第二个逗号
                e = d - 1
                right_num = 0
                left_num = 0
                while True:
                    if s[e] == '}':
                        right_num += 1
                    if s[e] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    e -= 1
                if left_num == 0:
                    e += 1
                f = e - 1  # 此时d的位置是算子最后一位符号
                while s[f] != ',':
                    f -= 1
                node.right = self.parse(s[f + 1:d])
                node.left = self.parse(s[a + 1:f])
                return node
            if self.operation_type_dic[name] == '3':
                # 定位第二个逗号
                left_num = 0
                right_num = 0
                b = len(s) - 2  # 此时b在倒数第二个位置，是}或者一个字母
                while True:
                    if s[b] == '}':
                        right_num += 1
                    if s[b] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b -= 1
                if left_num == 0:
                    b += 1
                c = b - 1  # 此时c的位置是算子最后一位符号
                while s[c] != ',':
                    c -= 1
                # 此时c的位置是第二个逗号的位置

                b_1 = c - 1
                left_num = 0
                right_num = 0
                while True:
                    if s[b_1] == '}':
                        right_num += 1
                    if s[b_1] == '{':
                        left_num += 1
                    if left_num == right_num:
                        break
                    b_1 -= 1
                if left_num == 0:
                    b_1 += 1
                c_1 = b_1 - 1
                while s[c_1] != ',':
                    c_1 -= 1  # 此时c_1的位置是第一个逗号

                node.right = self.parse(s[c + 1:len(s) - 1])
                node.left = self.parse(s[a + 1:c_1])
                node.middle = self.parse(s[c_1 + 1:c])
                return node
