# Copyright (c) 2022 Dai HBG

"""
AutoFormula的Cython版本
"""
import numpy as np
import sys
import datetime

sys.path.append('C:/Users/Administrator/Desktop/Repositories/High-Frequency-Predictor')
sys.path.append('C:/Users/HBG/Desktop/Repositories/High-Frequency-Predictor')

from DataLoader.DataLoader import Data
from AutoFormula.AutoTester import AutoTester, Stats
from AutoFormula.FormulaTree_cy import FormulaTree, Node, FormulaParser
from AutoFormula.SignalGenerator_cy import SignalGenerator


class AutoFormula_cy:
    def __init__(self, data):
        self.tree_generator = FormulaTree()
        self.operation = SignalGenerator(data=data)
        self.formula_parser = FormulaParser()
        self.auto_tester = AutoTester()

    def cal_formula(self, tree: FormulaTree, data_dic: dict,
                    return_type: str = 'signal') -> np.array:  # 递归计算公式树的值
        """
        :param tree: 需要计算的公式树
        :param data_dic: 原始数据的字典，可以通过字段读取对应的矩阵
        :param return_type: 返回值形式
        :return: 返回计算好的signal矩阵
        """
        if return_type == 'signal':
            if tree.variable_type == 'data':
                if type(tree.name) == int or type(tree.name) == float:
                    return tree.name  # 直接挂载在节点上，但是应该修改成需要数字的就直接返回数字
                return data_dic[tree.name].copy()  # 当前版本需要返回一个副本
            elif tree.variable_type == 'intra_data':
                if tree.num_1 is not None:
                    return data_dic[tree.name][:, tree.num_1, :].copy()
                else:
                    return data_dic[tree.name].copy()  # 如果没有数字就直接返回原本的数据
            else:
                if tree.operation_type == '1':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input, tree.num_1)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input, tree.num_1)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num_num':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input, tree.num_1, tree.num_2)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input, tree.num_1, tree.num_2)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num_num_num':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input, tree.num_1, tree.num_2,
                                                                               tree.num_3)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input, tree.num_1, tree.num_2,
                                                                               tree.num_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2':  # 此时需要判断有没有数字
                    if tree.num_1 is None:
                        input_1 = self.cal_formula(tree.left, data_dic, return_type)
                        input_2 = self.cal_formula(tree.right, data_dic, return_type)
                        if len(input_1.shape) == 2:
                            return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2)
                        elif len(input_1.shape) == 3:
                            return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2)
                    else:
                        # 暂时要求数字只能出现在后面
                        input_1 = self.cal_formula(tree.left, data_dic, return_type)
                        if len(input_1.shape) == 2:
                            return self.operation.operation_dic[tree.name + '_num_2d'](input_1, tree.num_1)
                        elif len(input_1.shape) == 3:
                            return self.operation.operation_dic[tree.name + '_num_3d'](input_1, tree.num_1)
                        else:
                            raise NotImplementedError('input shape is not right!')
                        # if tree.left is not None:
                        #     return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                        #                                                                     return_type),
                        #                                                    tree.num_1)
                        # else:
                        #     return self.operation.operation_dic[tree.name](tree.num_1,
                        #                                                    self.cal_formula(tree.right, data_dic,
                        #                                                                     return_type))
                if tree.operation_type == '2_num':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2_num_num':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1, tree.num_2)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1, tree.num_2)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2_num_num_num':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1,
                                                                               tree.num_2, tree.num_3)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1,
                                                                               tree.num_2, tree.num_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '3':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.middle, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type))
        if return_type == 'str':
            if tree.variable_type == 'data':
                return tree.name  # 返回字符串
            elif tree.variable_type == 'intra_data':  # 这里也需要判断是否有数字
                if tree.num_1 is not Nonr:
                    return '{' + tree.name + ',{}'.format(tree.num_1) + '}'
                else:
                    return '{' + tree.name + '}'
            else:
                if tree.operation_type == '1':
                    return tree.name + '{' + (self.cal_formula(tree.left, data_dic, return_type)) + '}'
                if tree.operation_type == '1_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + '}'
                if tree.operation_type == '1_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '1_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '2':  # 此时需要判断是否有数字
                    if tree.num_1 is not None:
                        return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                               self.cal_formula(tree.right, data_dic, return_type) + '}'
                    else:
                        if tree.left is not None:
                            return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                                   str(tree.num_1) + '}'
                        else:
                            return tree.name + '{' + str(tree.num_1) + ',' + \
                                   self.cal_formula(tree.right, data_dic, return_type) + '}'
                if tree.operation_type == '2_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + '}'
                if tree.operation_type == '2_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '2_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '3':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.middle, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + '}'

    def test_formula(self, formula: str, data: Data, start: int = 100,
                     end: int = 4600, shift: int = 1) -> (Stats, np.array):
        if type(formula) == str:
            formula = self.formula_parser.parse(formula)
        signal = self.cal_formula(formula, data.data_dic)  # 暂时为了方便，无论如何都计算整个回测区间的因子值
        return self.auto_tester.test(signal, data.ret, start=start, end=end, shift=shift), signal
