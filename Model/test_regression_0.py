# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:44:38 2022

@author: liuzenan
"""
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error
import os
import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim


class test_regression_0():
    def __init__(self, model_type, feature_name, future_time,data_train_loader,data_val_loader,data_test_loader):
        super().__init__()
        self.model_type = model_type
        self.feature_name = feature_name
        self.future_time = future_time
        self.data_train_loader = data_train_loader
        self.data_val_loader = data_val_loader
        self.data_test_loader = data_test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def test_train(self):
        model = torch.load("./model/best_val_model_pytorch_best_"+self.model_type+'_'+self.feature_name+"_"+str(self.future_time)+"min.pt")
        model.eval()
        all_targets = []
        all_predictions = []
        
        for inputs, targets in tqdm(self.data_train_loader):
            inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
            # Forward pass
            outputs = model(inputs)   
            predictions = torch.max(outputs,1)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.values.cpu().detach().numpy())
        all_targets = np.concatenate(all_targets)    
        all_predictions = np.concatenate(all_predictions)  
        all_predictions = all_predictions.reshape(len(all_targets))
        all_benchmark = np.array([np.mean(all_targets)]*len(all_targets))
        mse = (mean_squared_error(all_targets, all_predictions))*1000000
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))*1000000
        corr = np.corrcoef(all_targets, all_predictions)[0][1]
        mse_benchmark = (mean_squared_error(all_targets, all_benchmark))*1000000
        rmse_benchmark = np.sqrt(mean_squared_error(all_targets, all_benchmark))*1000000
        print(np.min(all_targets),np.max(all_targets))
        print(np.min(all_predictions),np.max(all_predictions))
        print('mean_squared_error:',mse,'       ',rmse)
        print('correlation:',corr)
        print('mean_squared_error_benchmark:',mse_benchmark,'      ',rmse_benchmark)
        print(1-mse/mse_benchmark)
    
    def test_val(self):
        model = torch.load("./model/best_val_model_pytorch_best_"+self.model_type+'_'+self.feature_name+"_"+str(self.future_time)+"min.pt")
        model.eval()
        all_targets = []
        all_predictions = []
        
        for inputs, targets in tqdm(self.data_val_loader):
            inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
            # Forward pass
            outputs = model(inputs)   
            predictions = torch.max(outputs,1)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.values.cpu().detach().numpy())
        all_targets = np.concatenate(all_targets)    
        all_predictions = np.concatenate(all_predictions)
        all_predictions = all_predictions.reshape(len(all_targets))
        all_benchmark = np.array([np.mean(all_targets)]*len(all_targets))
        mse = (mean_squared_error(all_targets, all_predictions))*1000000
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))*1000000
        corr = np.corrcoef(all_targets, all_predictions)[0][1]
        mse_benchmark = (mean_squared_error(all_targets, all_benchmark))*1000000
        rmse_benchmark = np.sqrt(mean_squared_error(all_targets, all_benchmark))*1000000
        print(np.min(all_targets),np.max(all_targets))
        print(np.min(all_predictions),np.max(all_predictions))
        print('mean_squared_error:',mse,'       ',rmse)
        print('correlation:',corr)
        print('mean_squared_error_benchmark:',mse_benchmark,'      ',rmse_benchmark)
        print(1-mse/mse_benchmark)

    def test_test(self):
        model = torch.load("./model/best_val_model_pytorch_best_"+self.model_type+'_'+self.feature_name+"_"+str(self.future_time)+"min.pt")
        model.eval()
        all_targets = []
        all_predictions = []
        
        for inputs, targets in tqdm(self.data_test_loader):
            inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
            # Forward pass
            outputs = model(inputs)   
            predictions = torch.max(outputs,1)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.values.cpu().detach().numpy())
        all_targets = np.concatenate(all_targets)    
        all_predictions = np.concatenate(all_predictions)
        all_predictions = all_predictions.reshape(len(all_targets))
        all_benchmark = np.array([np.mean(all_targets)]*len(all_targets))
        mse = (mean_squared_error(all_targets, all_predictions))*1000000
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))*1000000
        corr = np.corrcoef(all_targets, all_predictions)[0][1]
        mse_benchmark = (mean_squared_error(all_targets, all_benchmark))*1000000
        rmse_benchmark = np.sqrt(mean_squared_error(all_targets, all_benchmark))*1000000
        print(np.min(all_targets),np.max(all_targets))
        print(np.min(all_predictions),np.max(all_predictions))
        print('mean_squared_error:',mse,'       ',rmse)
        print('correlation:',corr)
        print('mean_squared_error_benchmark:',mse_benchmark,'      ',rmse_benchmark)
        print(1-mse/mse_benchmark)


