# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:14:26 2022

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
import numpy as np
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


class model_regression_best_parameters():
    def __init__(self, model_type, epochs, feature_name, future_time,data_train_loader,data_val_loader,para1,para2):
        super().__init__()
        self.model_type = model_type
        self.epochs = epochs
        self.feature_name = feature_name
        self.future_time = future_time
        self.data_train_loader = data_train_loader
        self.data_val_loader = data_val_loader
        self.para1 = para1
        self.para2 = para2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
    
    
    def batch_gd(self,model, criterion, optimizer):
        
        train_losses = np.zeros(self.epochs)
        test_losses = np.zeros(self.epochs)
        best_test_loss = np.inf
        best_test_epoch = 0
        
        for it in (range(self.epochs)):            
            all_predictions = []
            all_predictions_val = []
            
            model.train()
            t0 = datetime.now()
            train_loss = []
    
            for inputs, targets in tqdm(self.data_train_loader):
                inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
                optimizer.zero_grad()
                outputs = model(inputs)
                all_predictions.append(outputs.cpu().detach().numpy())
                loss = criterion(outputs, targets) * 1000000
                loss.backward()
                optimizer.step()
                train_loss.append((loss.item()))
            train_loss = np.mean(train_loss) # a little misleading
            all_predictions = np.concatenate(all_predictions)  
            print(np.min(all_predictions),np.max(all_predictions))
        
            model.eval()
            test_loss = []
            for inputs, targets in tqdm(self.data_val_loader):
                inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)      
                outputs = model(inputs)
                all_predictions_val.append(outputs.cpu().detach().numpy())
                loss = criterion(outputs, targets) * 1000000
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)
            all_predictions_val = np.concatenate(all_predictions_val)  
            print(np.min(all_predictions_val),np.max(all_predictions_val))
    
            train_losses[it] = train_loss
            test_losses[it] = test_loss
            
            if test_loss < best_test_loss:
                torch.save(model, "./model/best_val_model_pytorch_best_"+self.model_type+'_'+self.feature_name+"_"+str(self.future_time)+"min.pt")
                best_test_loss = test_loss
                best_test_epoch = it
            
    
            dt = datetime.now() - t0
            print(f'Epoch {it+1}/{self.epochs}, Train Loss: {train_loss:.4f}, \
              Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')
    
        return train_losses, test_losses

    def train_model(self):
        
        if self.model_type == 'LSTM':
            model = LSTM_model(**self.para1)
        elif self.model_type == 'stacked_LSTM':
            model = LSTM_stacked_model(**self.para1)
        elif self.model_type == 'MLP':
            model = MLP_model(**self.para1)
        elif self.model_type == 'LSTM-MLP':
            model = LSTM_MLP_model(**self.para1)
        elif self.model_type == 'CNN-LSTM':
            model = CNN_LSTM_model(**self.para1,device = self.device)
        model.to(self.device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), **self.para2)
        train_losses, val_losses = self.batch_gd(model, criterion, optimizer)
        return train_losses, val_losses