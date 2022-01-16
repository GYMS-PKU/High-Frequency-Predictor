# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:15:37 2022

@author: liuzenan
"""
from model_regression import LSTM_model
from model_regression import LSTM_stacked_model
from model_regression import MLP_model
from model_regression import LSTM_MLP_model
from model_regression import CNN_LSTM_model
import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 


class model_optuna():
    def __init__(self, model_type, epochs, feature_name, future_time,data_train_loader,data_val_loader):
        super().__init__()
        self.model_type = model_type
        self.epochs = epochs
        self.feature_name = feature_name
        self.future_time = future_time
        self.data_train_loader = data_train_loader
        self.data_val_loader = data_val_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)


    def batch_gd_optuna(self, model, criterion, optimizer):
    
        train_losses = np.zeros(self.epochs)
        test_losses = np.zeros(self.epochs)
        best_test_loss = np.inf
        best_test_epoch = 0
        
        for it in tqdm(range(self.epochs)):
            
            model.train()
            t0 = datetime.now()
            train_loss = []
    
            for inputs, targets in (self.data_train_loader):
                inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets) * 1000000
                loss.backward()
                optimizer.step()
                train_loss.append((loss.item()))
            train_loss = np.mean(train_loss) # a little misleading
        
            model.eval()
            test_loss = []
            for inputs, targets in (self.data_val_loader):
                inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)      
                outputs = model(inputs)
                loss = criterion(outputs, targets) * 1000000
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)
    
            train_losses[it] = train_loss
            test_losses[it] = test_loss
            
            if test_loss < best_test_loss:
                torch.save(model, "./model/best_val_model_pytorch_optuna_"+self.model_type+'_'+self.feature_name+"_"+str(self.future_time)+"min.pt")
                best_test_loss = test_loss
                best_test_epoch = it

        return best_test_loss

    def model_optuna_best(self,para1,para2):
        if self.model_type == 'LSTM':
            model = LSTM_model(**para1)
        elif self.model_type == 'stacked_LSTM':
            model = LSTM_stacked_model(**para1)
        elif self.model_type == 'MLP':
            model = MLP_model(**para1)
        elif self.model_type == 'LSTM-MLP':
            model = LSTM_MLP_model(**para1)
        elif self.model_type == 'CNN-LSTM':
            model = CNN_LSTM_model(**para1)
            
        model.to(self.device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), **para2)
        best_test_loss = self.batch_gd_optuna(model, criterion, optimizer)
        return best_test_loss
    
    def objective(self,trial):
        if self.model_type == 'CNN-LSTM':
            params1 = {
            'conv_filter_num':trial.suggest_int('conv_filter_num', 24, 48),
            'inception_num':trial.suggest_int('inception_num', 48, 96),
            'LSTM_num':trial.suggest_int('LSTM_num', 48, 96),
            'leaky_relu_alpha':trial.suggest_loguniform('leaky_relu_alpha', 1e-5, 0.1),
            'device':self.device,
              }
        elif self.model_type == 'LSTM':
            params1 = {
            'num_layers':2,
            'input_size':40,
            'hidden_size':trial.suggest_int('hidden_size', 60, 300),
                }
        elif self.model_type == 'stacked_LSTM':
            params1 = {
            'num_layers_s': 1,
            'input_size_s': 40,
            'hidden_size_s_1': trial.suggest_int('hidden_size_s_1', 50, 150),
            'hidden_size_s_2': trial.suggest_int('hidden_size_s_2', 100, 300),
            'hidden_size_s_3': trial.suggest_int('hidden_size_s_3', 200, 400),
                }
        elif self.model_type == 'MLP':
            params1 = {
            'inputsize':4000,
            'layer1':trial.suggest_int('layer1', 100, 800),
            'layer2':trial.suggest_int('layer2', 50, 150),
            'layer3':trial.suggest_int('layer3', 10, 60),
                }
        elif self.model_type == 'LSTM-MLP':
            params1 = {
            'num_layers':2,
            'input_size':40,
            'hidden_size':trial.suggest_int('hidden_size', 80, 200),
            'fc_size':trial.suggest_int('fc_size', 10, 50),
                }            
        params2 = {
        'lr':trial.suggest_loguniform('lr', 1e-5, 0.1),
           }
        return self.model_optuna_best(params1,params2)

    def getting_best_para(self):
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        study.optimize(lambda trial : self.objective(trial), n_trials=40, timeout = 48*60*60)
        print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
        return study.best_trial.params