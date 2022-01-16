# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 09:58:35 2022

@author: liuzenan
"""

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim


class LSTM_model(nn.Module):
    def __init__(self,num_layers, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_layers=num_layers,
                            input_size = self.input_size,
                            hidden_size = self.hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        
    def forward(self, our_data):
        """
        our_data: [batch_size, sequence_len, dims]:[256,100,40]
        """
        lstm_output, _ = self.lstm(our_data, None)
        output = self.fc(lstm_output)
        return output



class LSTM_stacked_model(nn.Module):
    def __init__(self,num_layers_s, input_size_s, hidden_size_s_1,hidden_size_s_2,hidden_size_s_3):
        super().__init__()        
        self.lstm1 = nn.LSTM(num_layers=num_layers_s,
                            input_size = input_size_s,
                            hidden_size = hidden_size_s_1)
        self.lstm2 = nn.LSTM(num_layers=num_layers_s,
                            input_size = hidden_size_s_1,
                            hidden_size = hidden_size_s_2)
        self.lstm3 = nn.LSTM(num_layers=num_layers_s,
                            input_size = hidden_size_s_2,
                            hidden_size = hidden_size_s_3)
        self.fc = nn.Linear(hidden_size_s_3,1)
        
    def forward(self, our_data):
        """
        our_data: [batch_size, sequence_len, dims]:[256,100,40]
        """
        lstm_output, _ = self.lstm1(our_data, None)
        lstm_output, _ = self.lstm2(lstm_output, None)
        lstm_output, _ = self.lstm3(lstm_output, None)
        output = self.fc(lstm_output)
        return output



    
class MLP_model(nn.Module):
    def __init__(self,inputsize,layer1,layer2,layer3):
        super().__init__()
        self.fc1 = nn.Linear(inputsize,layer1)
        self.fc2 = nn.Linear(layer1,layer2)
        self.fc3 = nn.Linear(layer2,layer3)
        self.fc4 = nn.Linear(layer3,1)
        
    def forward(self,our_data):
        """
        our_data: [batch_size,1,4000]:[256,1,4000]
        """
        mlp_output = nn.functional.relu(self.fc1(our_data))
        mlp_output = nn.functional.relu(self.fc2(mlp_output))
        mlp_output = nn.functional.relu(self.fc3(mlp_output))
        forecast_y = (self.fc4(mlp_output))
        return forecast_y
    


    
class LSTM_MLP_model(nn.Module):
    def __init__(self,num_layers, input_size, hidden_size, fc_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_layers=num_layers,
                            input_size = self.input_size,
                            hidden_size = self.hidden_size)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)
        
    def forward(self, our_data):
        """
        our_data: [batch_size, sequence_len, dims]:[256,100,40]
        """
        lstm_output, _ = self.lstm(our_data, None)
        output = nn.functional.relu(self.fc1(lstm_output))
        output = self.fc2(output)
        return output
    
    

class CNN_LSTM_model(nn.Module):
    def __init__(self,conv_filter_num,inception_num,leaky_relu_alpha,LSTM_num,device):
        super().__init__()
        self.conv_filter_num = conv_filter_num
        self.inception_num = inception_num
        self.leaky_relu_alpha = leaky_relu_alpha
        self.LSTM_num = LSTM_num
        self.device = device
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_filter_num, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(conv_filter_num),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(conv_filter_num),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(conv_filter_num),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_filter_num),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_filter_num),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_filter_num),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(conv_filter_num),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(conv_filter_num),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=conv_filter_num, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(conv_filter_num),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_filter_num, out_channels=inception_num, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(inception_num),
            nn.Conv2d(in_channels=inception_num, out_channels=inception_num, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(inception_num),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_filter_num, out_channels=inception_num, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(inception_num),
            nn.Conv2d(in_channels=inception_num, out_channels=inception_num, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(inception_num),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=conv_filter_num, out_channels=inception_num, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha),
            nn.BatchNorm2d(inception_num),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=3 * inception_num, hidden_size=LSTM_num, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(LSTM_num, 1)

    def forward(self, x):
        """
        our_data: [batch_size,1,100,40]:[256,1,100,40]
        """
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), self.LSTM_num).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.LSTM_num).to(self.device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        #forecast_y = torch.softmax(x, dim=1)
        forecast_y = x
        
        return forecast_y