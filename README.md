# High-Frequency-Predictor

这是一个用于测试高频数据预测力的Demo，主要的分为以下部分：

-- Model模块定义模型，所有和模型相关的功能实现都应该在Model模块下完成，其他代码调用该模块接口进行模型训练和预测。初始预测目标为分类问题，回归问题待定。初始模型为lightgbm；

-- DataProcess模块定义数据预处理方法，所有处理原始数据的方法都应该在该模块下完成；

-- Stats模块定义统计信息的方法，该模块是否集成于Model模块中待定，取决于是否需要统计其他统计量；

#### 流程

简化版本的流程应该依次为：

- 调用DataProcess模块得到np.array形式的输入和输出X，y；
- 调用Model模块训练并预测，打印统计信息；

#### 模块调用

-- Model模块调用为Model.fit(x,y,modelname='lgbm_regeression')