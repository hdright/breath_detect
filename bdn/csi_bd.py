#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib as plt
import torch
import os
import torch.nn as nn
import random
import time

# from bdn.loss import NMSE_cuda, NMSELoss, CosSimilarity, rho
from bdn.backbone import RegLSTM, BDCNN
from bdn.data_old import load_data
from bdn.data import load_data_from_txt
import matplotlib.pyplot as plt

model_path = "./model_save"
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def channel_visualization(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()

SEED = 42
print("seeding everything...")
seed_everything(SEED)
print("initializing parameters...")

class CNN_trainer():

    def __init__(self,
                 epochs,
                #  net,
                 #  feedbackbits=128,
                 BPMresol = 1.0,
                 breathEnd = 1,
                 batch_size=1,
                 learning_rate=1e-3,
                 lr_decay_freq=30,
                 lr_decay=0.1,
                 best_loss=100,
                 num_workers=0,
                 print_freq=25,
                 train_test_ratio=0.8):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay = lr_decay
        self.best_loss = best_loss
        self.num_workers = num_workers
        self.print_freq = print_freq
        self.train_test_ratio = train_test_ratio

        # 要估计的呼吸频率范围
        self.BPMresol = BPMresol
        self.breadthEnd = breathEnd
        resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
        bpmRange = np.arange(0, 60, BPMresol)
        noBpmPoints = len(bpmRange)  # 要估计的呼吸频率个数

        self.model = BDCNN(output_size=noBpmPoints)
        print("noBpmPoints: ", noBpmPoints)
        self.x_label = []
        self.y_label = []
        self.ys_label = []
        self.t_label = []

        if len(gpu_list.split(',')) > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()  # model.module
        else:
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_test = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # train_loader, test_loader, train_dataset, test_dataset, \
        # train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

        # self.train_loader, self.test_loader, self.train_dataset,        self.test_dataset, \
        # self.train_shuffle_loader, self.test_shuffle_loader,        \
        # self.train_shuffle_dataset, self.test_shuffle_dataset = \
        #     load_data_from_txt('./train_data', shuffle=True)
        self.train_shuffle_loader = load_data_from_txt(
                                                        Ridx = 0, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
                                                        BPMresolution = BPMresol, # 设置BPM分辨率
                                                        batch_size = batch_size, # 设置batch大小
                                                        shuffle = True, # 设置是否打乱数据
                                                        num_workers = 2, # 设置读取数据的线程数量
                                                )
        # self.test_loader = load_data_from_txt(
        #                                                 Ridx = 1, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
        #                                                 BPMresolution = BPMresol, # 设置BPM分辨率
        #                                                 batch_size = 1, # 设置batch大小
        #                                                 shuffle = False, # 设置是否打乱数据
        #                                                 num_workers = 2, # 设置读取数据的线程数量
        #                                         )


    def model_save(self,encodername, decodername):
        print('Saving model...')
#         encoderPATH = os.path.join(model_path, encodername)
#         decoderPATH = os.path.join(model_path, decodername)
#         try:
#             torch.save({'state_dict': self.model.encoder.state_dict(), }, encoderPATH)
#         except:
#             torch.save({'state_dict': self.model.module.encoder.state_dict(), }, encoderPATH)

#         try:
#             torch.save({'state_dict': self.model.decoder.state_dict(), }, decoderPATH)
#         except:
#             torch.save({'state_dict': self.model.module.decoder.state_dict(), }, decoderPATH)
# #         print('Model saved!')
#         self.best_loss = self.average_loss

    def model_load(self,encodername, decodername):
        encoderPATH = os.path.join(model_path, encodername)
        decoderPATH = os.path.join(model_path, decodername)

        # encoder_dict = torch.load(encoderPATH)
        # self.model.encoder.load_state_dict(encoder_dict['state_dict'], strict=False)  # type: ignore

        # decoder_dict = torch.load(decoderPATH)
        # self.model.decoder.load_state_dict(decoder_dict['state_dict'], strict=False)  # type: ignore

    def model_train(self):

        for epoch in range(self.epochs):
            print('========================')
            print('lr:%.4e' % self.optimizer.param_groups[0]['lr'])
            # train model
            self.model.train()
   
            # decay lr
            if epoch % self.lr_decay_freq == 0 and epoch > 0:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.lr_decay

            # training...
            for i, (x_in, label, cfg) in enumerate(self.train_shuffle_loader):
                self.optimizer.zero_grad()
                x_in = x_in.cuda()  # input [batch=1, 320, 600]
                x_in = torch.unsqueeze(x_in, 1) # [batch=1, 1, 320, 600]
                # print("x_in.shape:", x_in.shape)
                output = self.model(x_in) #.squeeze()
                # print("output.shape:", output.shape)
                label = label.cuda()
                # print("label.shape:", label.shape)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                if i % self.print_freq == 0:
                    with torch.no_grad():
                        pred_val = (torch.argsort(-output)[:, :cfg['Np']] * self.BPMresol).cpu()
                        pred_val.sort()
                        print("pred_val.shape:", pred_val.shape)
                        print("pred_val:", pred_val)
                        print("cfg['gt']:", cfg['gt'])
                        rmse = torch.sqrt(torch.mean((pred_val[:cfg['Np']] - cfg['gt']) ** 2))
                        print('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss:.4f}\t, RMSE {rmse:.4f}'.format(
                            epoch, i, len(self.train_shuffle_loader), loss=loss.item(), rmse=rmse.item()))
            self.model.eval()

            # evaluating...
            # self.total_loss = 0
            # self.total_rho = 0
            # start = time.time()
            # with torch.no_grad():

            #     for i, (x_in, label) in enumerate(self.test_shuffle_loader):
                    
            #         x_in = x_in.cuda()
            #         output = self.model(x_in).squeeze()
            #         label = label.cuda()
            #         print("output.shape:", output.shape)
            #         # 查看output的值
            #         print("output:", output)
            #         print("label.shape:", label.shape)
            #         print("label:", label)
            #         self.total_loss += self.criterion_test(output, label).item()
            #         # self.total_rho += self.criterion_rho(output,input).item()
            #         #print(rho(output,input), type(rho(output,input)))
            #         # self.total_rho += (rho(output,input))
                    
            #     end = time.time()
            #     t = end - start
            #     self.average_loss = self.total_loss / (i+1)
            #     # print("len(self.test_dataset):", len(self.test_dataset))
            #     # self.average_rho = self.total_rho / len(list(enumerate(self.test_loader)))
            #     self.x_label.append(epoch)
            #     self.y_label.append(self.average_loss)
            #     self.t_label.append(t)
            #     print('MSE %.4f time %.3f' % (self.average_loss, t))

        # for i, input in enumerate(self.test_loader): # visualize one sample
        #     if i == 3: # set shuffle = False to ensure the same sample each time
        #         ones = torch.ones(32,32)
        #         image1 = input[0].view(32,32)
        #         image1 = ones - image1
        #         image1 = image1.numpy()
        #         channel_visualization(image1)
        #         input = input.cuda()
        #         output = self.model(input)
        #         output = output.cpu()
        #         image2 = output[0].view(32,32)
        #         image2 = ones - image2
        #         image2 = image2.detach().numpy()
        #         channel_visualization(image2)

        # return self.x_label, self.y_label, sum(self.t_label)/len(self.t_label) # , self.ys_label


class LSTM_trainer():

    def __init__(self,
                 epochs,
                 net,
                 #  feedbackbits=128,
                 inp_dim=2, # regLSTM
                 out_dim=1, # regLSTM
                 mid_dim=40, # regLSTM
                 linear_dim=250, # regLSTM
                 mid_layers=2, # regLSTM
                 batch_size=30,
                 learning_rate=1e-3,
                 lr_decay_freq=30,
                 lr_decay=0.1,
                 best_loss=100,
                 num_workers=0,
                 print_freq=100,
                 train_test_ratio=0.8):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay = lr_decay
        self.best_loss = best_loss
        self.num_workers = num_workers
        self.print_freq = print_freq
        self.train_test_ratio = train_test_ratio
        # parameters for data
        # self.feedback_bits = feedbackbits
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.linear_dim = linear_dim
        self.mid_layers = mid_layers

        self.model = eval(net)(inp_dim=self.inp_dim, out_dim=self.out_dim, mid_dim=self.mid_dim,
                               linear_dim=self.linear_dim, mid_layers=self.mid_layers)
        self.x_label = []
        self.y_label = []
        self.ys_label = []
        self.t_label = []

        if len(gpu_list.split(',')) > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()  # model.module
        else:
            self.model = self.model.cuda()

        # self.criterion = NMSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        # self.criterion_test = NMSELoss(reduction='sum')
        self.criterion_test = nn.MSELoss()
        # self.criterion_rho = CosSimilarity(reduction='mean')
        # self.criterion_test_rho = CosSimilarity(reduction='sum')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # train_loader, test_loader, train_dataset, test_dataset, \
        # train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

        self.train_loader, self.test_loader, self.train_dataset,        self.test_dataset, self.train_shuffle_loader, self.test_shuffle_loader,        self.train_shuffle_dataset, self.test_shuffle_dataset = \
            load_data('./train_data', shuffle=True)

    def model_save(self,encodername, decodername):
        print('Saving model...')
        encoderPATH = os.path.join(model_path, encodername)
        decoderPATH = os.path.join(model_path, decodername)
        try:
            torch.save({'state_dict': self.model.encoder.state_dict(), }, encoderPATH)
        except:
            torch.save({'state_dict': self.model.module.encoder.state_dict(), }, encoderPATH)

        try:
            torch.save({'state_dict': self.model.decoder.state_dict(), }, decoderPATH)
        except:
            torch.save({'state_dict': self.model.module.decoder.state_dict(), }, decoderPATH)
#         print('Model saved!')
        self.best_loss = self.average_loss

    def model_load(self,encodername, decodername):
        encoderPATH = os.path.join(model_path, encodername)
        decoderPATH = os.path.join(model_path, decodername)

        encoder_dict = torch.load(encoderPATH)
        self.model.encoder.load_state_dict(encoder_dict['state_dict'], strict=False)  # type: ignore

        decoder_dict = torch.load(decoderPATH)
        self.model.decoder.load_state_dict(decoder_dict['state_dict'], strict=False)  # type: ignore

    def model_train(self):

        for epoch in range(self.epochs):
            print('========================')
            print('lr:%.4e' % self.optimizer.param_groups[0]['lr'])
            # train model
            self.model.train()
   
            # decay lr
            if epoch % self.lr_decay_freq == 0 and epoch > 0:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.lr_decay

            # training...
            for i, (x_in, label) in enumerate(self.train_shuffle_loader):
                x_in = x_in.cuda()  # input [batch=32,2,16,32]
                output = self.model(x_in).squeeze()
                label = label.cuda()
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f}\t'.format(
                        epoch, i, len(self.train_loader), loss=loss.item()))
            self.model.eval()

            # evaluating...
            self.total_loss = 0
            self.total_rho = 0
            start = time.time()
            with torch.no_grad():

                for i, (x_in, label) in enumerate(self.test_shuffle_loader):
                    
                    x_in = x_in.cuda()
                    output = self.model(x_in).squeeze()
                    label = label.cuda()
                    print("output.shape:", output.shape)
                    # 查看output的值
                    print("output:", output)
                    print("label.shape:", label.shape)
                    print("label:", label)
                    self.total_loss += self.criterion_test(output, label).item()
                    # self.total_rho += self.criterion_rho(output,input).item()
                    #print(rho(output,input), type(rho(output,input)))
                    # self.total_rho += (rho(output,input))
                    
                end = time.time()
                t = end - start
                self.average_loss = self.total_loss / (i+1)
                # print("len(self.test_dataset):", len(self.test_dataset))
                # self.average_rho = self.total_rho / len(list(enumerate(self.test_loader)))
                self.x_label.append(epoch)
                self.y_label.append(self.average_loss)
                self.t_label.append(t)
                print('MSE %.4f time %.3f' % (self.average_loss, t))

        # for i, input in enumerate(self.test_loader): # visualize one sample
        #     if i == 3: # set shuffle = False to ensure the same sample each time
        #         ones = torch.ones(32,32)
        #         image1 = input[0].view(32,32)
        #         image1 = ones - image1
        #         image1 = image1.numpy()
        #         channel_visualization(image1)
        #         input = input.cuda()
        #         output = self.model(input)
        #         output = output.cpu()
        #         image2 = output[0].view(32,32)
        #         image2 = ones - image2
        #         image2 = image2.detach().numpy()
        #         channel_visualization(image2)

        return self.x_label, self.y_label, sum(self.t_label)/len(self.t_label) # , self.ys_label
