#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
# 导入inception
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, BasicConv2d
# import torch.utils.model_zoo as model_zoo


# def inception_v3(pretrained=False, **kwargs):
#     r"""Inception v3 model architecture from
#     `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         if 'transform_input' not in kwargs:
#             kwargs['transform_input'] = True
#         model = BDInception3(**kwargs)
#         # model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
#         return model

#     return BDInception3(**kwargs)
class InceptionAux(nn.Module):

    def __init__(self, in_channels, output_size, input_sample=640):
        super(InceptionAux, self).__init__()
        self.input_sample = input_sample
        assert self.input_sample in [180, 640]
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        if self.input_sample == 640:
            self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        else:
            self.conv1 = BasicConv2d(128, 768, kernel_size=(4, 5))
        self.conv1.stddev = torch.tensor(0.01)
        self.fc = nn.Linear(768, output_size)
        self.fc.stddev = torch.tensor(0.001)

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3, padding=1)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x

class BDInception3(nn.Module):

    def __init__(self, input_sample=640, output_size=600, aux_logits=False):
        super(BDInception3, self).__init__()
        self.aux_logits = aux_logits
        self.input_sample = input_sample
        # self.transform_input = transform_input
        assert self.input_sample in [180, 640]
        if input_sample == 640:
            self.Conv2d_1a_3x3 = BasicConv2d(2, 32, kernel_size=40, stride=(2, 4))
            self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
            self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
            self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
            self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
            self.Mixed_5b = InceptionA(192, pool_features=32)
            self.Mixed_5c = InceptionA(256, pool_features=64)
            self.Mixed_5d = InceptionA(288, pool_features=64)
            self.Mixed_6a = InceptionB(288)
            self.Mixed_6b = InceptionC(768, channels_7x7=128)
            self.Mixed_6c = InceptionC(768, channels_7x7=160)
            self.Mixed_6d = InceptionC(768, channels_7x7=160)
            self.Mixed_6e = InceptionC(768, channels_7x7=192)
            if aux_logits:
                self.AuxLogits = InceptionAux(768, output_size)
            self.Mixed_7a = InceptionD(768)
            self.Mixed_7b = InceptionE(1280)
            self.Mixed_7c = InceptionE(2048)
            self.fc = nn.Linear(2048, output_size)
        else:
            self.Conv2d_1a_3x3 = BasicConv2d(2, 32, kernel_size=(30, 40), stride=(1, 4))
            self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
            self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
            self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
            self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
            self.Mixed_5b = InceptionA(192, pool_features=32)
            self.Mixed_5c = InceptionA(256, pool_features=64)
            self.Mixed_5d = InceptionA(288, pool_features=64)
            self.Mixed_6a = InceptionB(288)
            self.Mixed_6b = InceptionC(768, channels_7x7=128)
            self.Mixed_6c = InceptionC(768, channels_7x7=160)
            self.Mixed_6d = InceptionC(768, channels_7x7=160)
            self.Mixed_6e = InceptionC(768, channels_7x7=192)
            if aux_logits:
                self.AuxLogits = InceptionAux(768, output_size)
            self.Mixed_7a = InceptionD(768)
            self.Mixed_7b = InceptionE(1280)
            self.Mixed_7c = InceptionE(2048)
            self.fc = nn.Linear(2048, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # if self.transform_input: # 1
        #     x = x.clone()
        #     x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        #     x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        #     x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        if self.input_sample == 640:
            # 320 x 600 x 2
            x = self.Conv2d_1a_3x3(x)
            # 141 x 141 x 32
            x = self.Conv2d_2a_3x3(x)
            # 139 x 139 x 32
            x = self.Conv2d_2b_3x3(x)
            # 139 x 139 x 64
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 69 x 69 x 64
            x = self.Conv2d_3b_1x1(x)
            # 69 x 69 x 80
            x = self.Conv2d_4a_3x3(x)
            # 67 x 67 x 192
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 33 x 33 x 192
            x = self.Mixed_5b(x)
            # 33 x 33 x 256
            x = self.Mixed_5c(x)
            # 33 x 33 x 288
            x = self.Mixed_5d(x)
            # 33 x 33 x 288
            x = self.Mixed_6a(x)
            # 16 x 16 x 768
            x = self.Mixed_6b(x)
            # 16 x 16 x 768
            x = self.Mixed_6c(x)
            # 16 x 16 x 768
            x = self.Mixed_6d(x)
            # 16 x 16 x 768
            x = self.Mixed_6e(x)
            # 16 x 16 x 768
            if self.training and self.aux_logits:
                aux = self.AuxLogits(x)
            # 16 x 16 x 768
            x = self.Mixed_7a(x)
            # 7 x 7 x 1280
            x = self.Mixed_7b(x)
            # 7 x 7 x 2048
            x = self.Mixed_7c(x)
            # 7 x 7 x 2048
            x = F.avg_pool2d(x, kernel_size=7)
            # 1 x 1 x 2048
            x = F.dropout(x, training=self.training)
            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)
            # 2048
            x = self.fc(x)
        else:
            # 90 x 600 x 2
            x = self.Conv2d_1a_3x3(x)
            # 61 x 141 x 32
            x = self.Conv2d_2a_3x3(x)
            # 59 x 139 x 32
            x = self.Conv2d_2b_3x3(x)
            # 59 x 139 x 64
            x = F.max_pool2d(x, kernel_size=(1, 3), stride=(1, 2))
            # 59 x 69 x 64
            x = self.Conv2d_3b_1x1(x)
            # 59 x 69 x 80
            x = self.Conv2d_4a_3x3(x)
            # 57 x 67 x 192
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 28 x 33 x 192
            x = self.Mixed_5b(x)
            # 28 x 33 x 256
            x = self.Mixed_5c(x)
            # 28 x 33 x 288
            x = self.Mixed_5d(x)
            # 28 x 33 x 288
            x = self.Mixed_6a(x)
            # 13 x 16 x 768
            x = self.Mixed_6b(x)
            # 13 x 16 x 768
            x = self.Mixed_6c(x)
            # 13 x 16 x 768
            x = self.Mixed_6d(x)
            # 13 x 16 x 768
            x = self.Mixed_6e(x)
            # 13 x 16 x 768
            if self.training and self.aux_logits:
                aux = self.AuxLogits(x)
            # 13 x 16 x 768
            x = self.Mixed_7a(x)
            # 6 x 7 x 1280
            x = self.Mixed_7b(x)
            # 6 x 7 x 2048
            x = self.Mixed_7c(x)
            # 6 x 7 x 2048
            x = F.avg_pool2d(x, kernel_size=(6, 7))
            # 1 x 1 x 2048
            x = F.dropout(x, training=self.training)
            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)
            # 2048
            x = self.fc(x)
        # 600 (output_size)
        if self.training and self.aux_logits:
            return x, aux
        return x


class BDCNN(nn.Module):
    def __init__(self, input_sample=640, output_size=600):
        super(BDCNN, self).__init__()   # 继承__init__功能
        if input_sample in [640,180640]:
            ## 第一层卷积
            self.conv1 = nn.Sequential(
                # 输入[2,320,600]
                nn.Conv2d(
                    in_channels=2,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(22,35),    # 22*35的卷积核，相当于过滤器
                    stride=(10,5),         # 卷积核在图上滑动，每隔个扫一次
                    padding=(6,0),        # 给图外边补上0
                ),
                nn.Conv2d(
                    in_channels=64,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(7,15),    # 7*15的卷积核，相当于过滤器
                    stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                    padding=(2,2),        # 给图外边补上0
                ),
                # 经过卷积层 输出[64,30,104] 传入BN、池化层
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3))   # 经过池化 输出[64,10,34] 传入下一个卷积
            )
            ## 第二层卷积
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,    # 同上
                    out_channels=128,
                    kernel_size=(3,7),
                    stride=1,
                    padding=0
                ),
                nn.Conv2d(
                    in_channels=128,    # 同上
                    out_channels=256,
                    kernel_size=(3,7),
                    stride=1,
                    # padding=(0,1)
                    padding=0
                ),
                # 经过卷积 输出[256, 6, 24] 传入BN、池化层
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,3))  # 经过池化 输出[256,6,8] 传入输出层
            )
            ## 输出层
            # self.linear1 = nn.Sequential(nn.Linear(in_features=256*6*8, out_features=8192), 
            self.linear1 = nn.Sequential(nn.Linear(in_features=256*6*7, out_features=8192), 
                                            nn.ReLU(),
                                            nn.Dropout(0.5)
                                            )
            self.linear2 = nn.Sequential(nn.Linear(in_features=8192, out_features=4096), 
                                            nn.ReLU(),
                                            nn.Dropout(0.5)
                                            )
            self.linear3 = nn.Linear(in_features=4096, out_features=output_size)
            # softmax输出分类
            # self.softmax = nn.Softmax(dim=1)
        else:
            ## 第一层卷积
            self.conv1 = nn.Sequential(
                # 输入[1,320,600]
                nn.Conv2d(
                    in_channels=2,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(18,35),    # 16*35的卷积核，相当于过滤器
                    stride=(4,5),         # 卷积核在图上滑动，每隔个扫一次
                    padding=(6,0),        # 给图外边补上0
                ),
                nn.Conv2d(
                    in_channels=64,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(7,15),    # 16*35的卷积核，相当于过滤器
                    stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                    padding=(2,2),        # 给图外边补上0
                ),
                # 经过卷积层 输出[32,32,104] 传入BN、池化层
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3))   # 经过池化 输出[32,16,52] 传入下一个卷积
            )
            ## 第二层卷积
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,    # 同上
                    out_channels=128,
                    kernel_size=(3,7),
                    stride=1,
                    padding=0
                ),
                nn.Conv2d(
                    in_channels=128,    # 同上
                    out_channels=256,
                    kernel_size=(3,7),
                    stride=1,
                    padding=(0,1)
                ),
                # 经过卷积 输出[64, 14, 44] 传入BN、池化层
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(4,3))  # 经过池化 输出[64,14,22] 传入输出层
            )
            ## 输出层
            self.linear1 = nn.Sequential(nn.Linear(in_features=256*4*8, out_features=8192), 
                                            nn.ReLU(),
                                            nn.Dropout(0.5)
                                            )
            self.linear2 = nn.Sequential(nn.Linear(in_features=8192, out_features=4096), 
                                            nn.ReLU(),
                                            nn.Dropout(0.5)
                                            )
            self.linear3 = nn.Linear(in_features=4096, out_features=output_size) # batch_size * output_size
            # softmax输出分类
            # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # [batch, 64,14,22]
        x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 64*14*22]
        linear_o1 = self.linear1(x)     # 输出[50,10]
        linear_o2 = self.linear2(linear_o1)     # 输出[50,10]
        linear_o3 = self.linear3(linear_o2)     # 输出[50,10]
        # output = self.softmax(linear_o3) # 输出[50,10]
        return linear_o3

class BDCNN_ND(nn.Module): # 无dropout
    def __init__(self, input_sample=640, output_size=600):
        super(BDCNN_ND, self).__init__()   # 继承__init__功能
        if input_sample == 640:
            ## 第一层卷积
            self.conv1 = nn.Sequential(
                # 输入[2,320,600]
                nn.Conv2d(
                    in_channels=2,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(22,35),    # 22*35的卷积核，相当于过滤器
                    stride=(10,5),         # 卷积核在图上滑动，每隔个扫一次
                    padding=(6,0),        # 给图外边补上0
                ),
                nn.Conv2d(
                    in_channels=64,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(7,15),    # 7*15的卷积核，相当于过滤器
                    stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                    padding=(2,2),        # 给图外边补上0
                ),
                # 经过卷积层 输出[64,30,104] 传入池化层
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3))   # 经过池化 输出[64,10,34] 传入下一个卷积
            )
            ## 第二层卷积
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,    # 同上
                    out_channels=128,
                    kernel_size=(3,7),
                    stride=1,
                    padding=0
                ),
                nn.Conv2d(
                    in_channels=128,    # 同上
                    out_channels=256,
                    kernel_size=(3,7),
                    stride=1,
                    # padding=(0,1)
                    padding=0
                ),
                # 经过卷积 输出[256, 6, 24] 传入池化层
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,3))  # 经过池化 输出[256,6,8] 传入输出层
            )
            ## 输出层
            # self.linear1 = nn.Sequential(nn.Linear(in_features=256*6*8, out_features=8192), 
            self.linear1 = nn.Sequential(nn.Linear(in_features=256*6*7, out_features=8192), 
                                            nn.ReLU(),
                                            # nn.Dropout(0.5)
                                            )
            self.linear2 = nn.Sequential(nn.Linear(in_features=8192, out_features=4096), 
                                            nn.ReLU(),
                                            # nn.Dropout(0.5)
                                            )
            self.linear3 = nn.Linear(in_features=4096, out_features=output_size)
            # softmax输出分类
            # self.softmax = nn.Softmax(dim=1)
        else:
            ## 第一层卷积
            self.conv1 = nn.Sequential(
                # 输入[1,320,600]
                nn.Conv2d(
                    in_channels=2,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(18,35),    # 16*35的卷积核，相当于过滤器
                    stride=(4,5),         # 卷积核在图上滑动，每隔个扫一次
                    padding=(6,0),        # 给图外边补上0
                ),
                nn.Conv2d(
                    in_channels=64,    # 输入图片的高度
                    out_channels=64,  # 输出图片的高度
                    kernel_size=(7,15),    # 16*35的卷积核，相当于过滤器
                    stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                    padding=(2,2),        # 给图外边补上0
                ),
                # 经过卷积层 输出[32,32,104] 传入池化层
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3))   # 经过池化 输出[32,16,52] 传入下一个卷积
            )
            ## 第二层卷积
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,    # 同上
                    out_channels=128,
                    kernel_size=(3,7),
                    stride=1,
                    padding=0
                ),
                nn.Conv2d(
                    in_channels=128,    # 同上
                    out_channels=256,
                    kernel_size=(3,7),
                    stride=1,
                    padding=(0,1)
                ),
                # 经过卷积 输出[64, 14, 44] 传入池化层
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(4,3))  # 经过池化 输出[64,14,22] 传入输出层
            )
            ## 输出层
            self.linear1 = nn.Sequential(nn.Linear(in_features=256*4*8, out_features=8192), 
                                            nn.ReLU(),
                                            # nn.Dropout(0.5)
                                            )
            self.linear2 = nn.Sequential(nn.Linear(in_features=8192, out_features=4096), 
                                            nn.ReLU(),
                                            # nn.Dropout(0.5)
                                            )
            self.linear3 = nn.Linear(in_features=4096, out_features=output_size)
            # softmax输出分类
            # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # [batch, 64,14,22]
        x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 64*14*22]
        linear_o1 = self.linear1(x)     # 输出[50,10]
        linear_o2 = self.linear2(linear_o1)     # 输出[50,10]
        linear_o3 = self.linear3(linear_o2)     # 输出[50,10]
        # output = self.softmax(linear_o2) # 输出[50,10]
        return linear_o3

class BDCNNold(nn.Module):
    def __init__(self, input_sample=320, output_size=600):
        super(BDCNNold, self).__init__()   # 继承__init__功能
        ## 第一层卷积
        if input_sample == 320 or input_sample == 90320:
            padding00 = 3
            kernel00 = 16
            stride00 = 10
            pool00 = 2
        else:
            padding00 = 6
            kernel00 = 12
            stride00 = 6
            pool00 = 1
        self.conv1 = nn.Sequential(
            # 输入[1,320,600]
            nn.Conv2d(
                in_channels=1,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=(kernel00,35),    # 16*35的卷积核，相当于过滤器
                stride=(stride00,5),         # 卷积核在图上滑动，每隔个扫一次
                padding=(padding00,0),        # 给图外边补上0
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
            nn.MaxPool2d(kernel_size=(pool00, 2))   # 经过池化 输出[32,16,52] 传入下一个卷积
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

