#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class BDCNN(nn.Module):
    def __init__(self, output_size=600):
        super(BDCNN, self).__init__()   # 继承__init__功能
        ## 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,320,600]
            nn.Conv2d(
                in_channels=1,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=(16,35),    # 16*35的卷积核，相当于过滤器
                stride=(10,5),         # 卷积核在图上滑动，每隔个扫一次
                padding=(3,0),        # 给图外边补上0
            ),
            nn.Conv2d(
                in_channels=16,    # 输入图片的高度
                out_channels=32,  # 输出图片的高度
                kernel_size=(7,15),    # 16*35的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=(3,2),        # 给图外边补上0
            ),
            # 经过卷积层 输出[32,32,104] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 经过池化 输出[32,16,52] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,    # 同上
                out_channels=64,
                kernel_size=(5,7),
                stride=1,
                padding=2
            ),
            nn.Conv2d(
                in_channels=64,    # 同上
                out_channels=64,
                kernel_size=(3,7),
                stride=1,
                padding=0
            ),
            # 经过卷积 输出[64, 14, 44] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2))  # 经过池化 输出[64,14,22] 传入输出层
        )
        ## 输出层
        self.linear1 = nn.Sequential(nn.Linear(in_features=64*14*22, out_features=6144), 
                                        nn.ReLU())
        self.linear2 = nn.Linear(in_features=6144, out_features=output_size)
        # softmax输出分类
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # [batch, 64,14,22]
        x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 64*14*22]
        linear_o1 = self.linear1(x)     # 输出[50,10]
        linear_o2 = self.linear2(linear_o1)     # 输出[50,10]
        # output = self.softmax(linear_o2) # 输出[50,10]
        return linear_o2

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
            # nn.Tanh(),
            nn.ReLU(),
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

