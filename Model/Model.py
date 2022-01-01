# Copyright (c) 2021 Dai HBG

"""
Model是若干个定义了多种标准化以及结构化的模型框架，可以直接调用用于模型拟合

开发日志：
"""
import sys
from sklearn import linear_model
from copy import deepcopy
import numpy as np
import pickle
import lightgbm as lgb


class MyBoostingModel:  # 为了统一接口，需要定义fit方法
    def __init__(self, lasso, lgbm):
        self.lasso = lasso
        self.lgbm = lgbm

    def predict(self, x):
        return self.lasso.predict(x) + self.lgbm.predict(x)


class Model:
    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, x_test=None, y_test=None, model=None, param=None):
        """
        :param x_train: 训练集x
        :param y_train: 训练集y
        :param x_test: 测试集x
        :param y_test: 测试集y
        :param model: 结构化模型名字
        :param param: 该模型对应的参数
        :return:
        """
        if model is None or model == 'Lasso':  # 默认使用Lasso
            self.model = linear_model.Lasso(alpha=5e-4)
            print('there are {} factors'.format(x_train.shape[1]))
            self.model.fit(x_train, y_train)
            print('{} factors have been selected'.format(np.sum(self.model.coef_ != 0)))
            print('training corr is {:.4f}'.format(np.corrcoef(y_train, self.model.predict(x_train))[0, 1]))
            if x_test is not None:
                print('testing corr is {:.4f}'.format(np.corrcoef(y_test, self.model.predict(x_test))[0, 1]))
        if model == 'lgbm_regression':  # 这里是一个回归模型，
            params = {'num_leaves': 20, 'min_data_in_leaf': 50, 'objective': 'regression', 'max_depth': 6,
                      'learning_rate': 0.05, "min_sum_hessian_in_leaf": 6,
                      "boosting": "gbdt", "feature_fraction": 0.9, "bagging_freq": 1, "bagging_fraction": 0.7,
                      "bagging_seed": 11, "lambda_l1": 2, "verbosity": 1, "nthread": -1,
                      'metric': 'mae', "random_state": 2019}  # 'device': 'gpu'}
            num_round = 100
            trn_data = lgb.Dataset(x_train, label=y_train)
            val_data = lgb.Dataset(x_test, label=y_test)
            self.model = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=20)
        if model == 'lgbm_classification':  # 这里是分类模型，注意之类的params没有定义好
            params = {'num_leaves': 20, 'min_data_in_leaf': 50, 'objective': 'multiclass', 'num_class':3,
                      'max_depth': 6,
                      'learning_rate': 0.05, "min_sum_hessian_in_leaf": 6,
                      "boosting": "gbdt", "feature_fraction": 0.9, "bagging_freq": 1, "bagging_fraction": 0.7,
                      "bagging_seed": 11, "lambda_l1": 2, "verbosity": 1, "nthread": -1,
                      'metric': 'multi_logloss', "random_state": 2019}  # 'device': 'gpu'}
            num_round = 100
            trn_data = lgb.Dataset(x_train, label=y_train)
            val_data = lgb.Dataset(x_test, label=y_test)
            self.model = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=20)

        if model == 'Lasso_lgbm_boosting':  # 这个是boosting模型，目前适用于回归
            model_1 = linear_model.Lasso(alpha=5e-4)
            model_1.fit(x_train, y_train)
            params = {'num_leaves': 20, 'min_data_in_leaf': 50, 'objective': 'regression', 'max_depth': 6,
                      'learning_rate': 0.05, "min_sum_hessian_in_leaf": 6,
                      "boosting": "gbdt", "feature_fraction": 0.9, "bagging_freq": 1, "bagging_fraction": 0.7,
                      "bagging_seed": 11, "lambda_l1": 2, "verbosity": 1, "nthread": -1,
                      'metric': 'mae', "random_state": 2019}  # 'device': 'gpu'}
            num_round = 100
            trn_data = lgb.Dataset(x_train, label=y_train - model_1.predict(x_train))
            val_data = lgb.Dataset(x_test, label=y_test - model_1.predict(x_test))
            model_2 = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=20)
            self.model = MyBoostingModel(model_1, model_2)

    def model_fit(self, model=None):  # 传入自定义模型进行训练
        if model is None:
            self.model = linear_model.Lasso(alpha=6e-4)
        else:
            self.model = model
        pass

    def dump_model(self, model_name):  # 保存模型
        with open('F:/Documents/AutoFactoryData/Model/{}.pkl'.format(model_name), 'wb') as file:
            pickle.dump(self.model, file)
