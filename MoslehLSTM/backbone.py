#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size=3000, hidden_layer_size=100, output_size=1):
        """
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size:隐层的数目
        :param output_size: 输出的个数
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  # shape: (n_layers, batch, hidden_size)
                       torch.zeros(1, 1, self.hidden_layer_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # =self.linear(lstm_out[:, -1, :])
        predictions = self.sigmoid(linear_out)
        return predictions
    
class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, linear_dim, mid_layers,batch=True):
        super(RegLSTM, self).__init__()
 
        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers, bidirectional=True)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(2*mid_dim*1500, linear_dim),
            nn.Tanh(),
            nn.Linear(linear_dim, out_dim),
        )  # regression
 
    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)
        # print(y.shape)
        batch_size, seq_len, hid_dim = y.shape
        y = y.reshape(-1, hid_dim*seq_len)
        y = self.reg(y)
        y = y.reshape(batch_size, -1)
        return y

