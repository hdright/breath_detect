#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib as plt
import torch
import os
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import random
import time
import pickle
# findpeak
from scipy.signal import find_peaks

# from bdn.loss import NMSE_cuda, NMSELoss, CosSimilarity, rho
from bdn.backbone import RegLSTM, BDCNN
from bdn.data_old import load_data
from bdn.data import load_data_from_txt, save_data_to_txt
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
                 train_now=True,
                 no_sample=320, # 设置读取哪种txt文件，90样本或者320样本
                 BPMresol = 1.0,
                 breathEnd = 1,
                 batch_size=1,
                 learning_rate=1e-3,
                 lr_decay_freq=30,
                 lr_decay=0.1,
                #  best_loss=100,
                 num_workers=0,
                 print_freq=25,
                 train_test_ratio=0.8):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay = lr_decay
        # self.best_loss = best_loss
        self.num_workers = num_workers
        self.print_freq = print_freq
        self.train_test_ratio = train_test_ratio
        self.model_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        self.no_sample = no_sample  # 读取的样本数
        # 要估计的呼吸频率范围
        self.BPMresol = BPMresol
        self.breadthEnd = breathEnd
        # resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
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
        if train_now:
            # train002009 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1.pkl'
            # train002009 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig0.1.pkl'
            # train002009 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig10.pkl'
            # train002009 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig100.pkl'
            train002009 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1000.pkl'
            train001 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90.pkl'
            train90320 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90320.pkl'
            train_pkl = train002009
            if os.path.exists(train_pkl):
                print('Loading train_shuffle_loader...')
                with open(train_pkl, 'rb') as f:
                    self.train_shuffle_loader = pickle.load(f)
            else:
                self.train_shuffle_loader = load_data_from_txt(
                                                                Ridx = 0, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
                                                                no_sample=no_sample, # 设置读取哪种txt文件，90样本或者320样本
                                                                BPMresolution = BPMresol, # 设置BPM分辨率
                                                                batch_size = batch_size, # 设置batch大小
                                                                shuffle = True, # 设置是否打乱数据
                                                                num_workers = 2, # 设置读取数据的线程数量
                                                        )
                with open(train_pkl, 'wb') as f:
                    pickle.dump(self.train_shuffle_loader, f)



    def model_save(self, name = "BDCNN", path = "./model_save"):
        print('Saving model...')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = name + '_' + self.model_time + '.pkl'
        modelPATH = os.path.join(path, self.model_time, model_name)
        torch.save({'state_dict': self.model.state_dict(), }, modelPATH)
        # 保存模型结构
        with open(os.path.join(path, self.model_time, model_name + '.txt'), 'w') as f:
            print("epochs: ", self.epochs, file=f)
            print("lr: ", self.learning_rate, file=f)
            print("lr_decay_freq: ", self.lr_decay_freq, file=f)
            print("lr_decay: ", self.lr_decay, file=f)
            print("batch_size: ", self.batch_size, file=f)
            print("best_loss: ", self.best_loss, file=f)
            print("best_rmse: ", self.best_rmse, file=f)
            print(self.model, file=f)
        print('Model saved!')
        

    def model_load(self, name, path = "./model_save"):
        print('Loading model...')
        modelPATH = os.path.join(path, name)
        model_dict = torch.load(modelPATH)
        self.model.load_state_dict(model_dict['state_dict'], strict=False)  # type: ignore


    def model_train(self):
        writer = SummaryWriter(log_dir='./model_save/'+self.model_time+'/')
        for epoch in range(self.epochs):
            print('========================')
            print('lr:%.4e' % self.optimizer.param_groups[0]['lr'])
            # train model
            self.model.train()
   
            # decay lr
            if epoch % self.lr_decay_freq == 0 and epoch > 0:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.lr_decay

            # training...
            se_total = torch.zeros(1)
            se_001 = torch.zeros(1)
            Np_total = torch.zeros(1)
            loss_total = torch.zeros(1)
            len_train_loader = len(self.train_shuffle_loader)
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
                        # print("type(output):", type(output))
                        # print("type(cfg['Np']):", type(cfg['Np']))
                        # print("cfg['Np'].numpy():", cfg['Np'].numpy()) 
                        if self.batch_size == 1:
                            # if batch size = 4, cfg['Np'].numpy(): [1 1 1 1], TypeError: only integer scalar arrays can be converted to a scalar index
                            output = output.squeeze() # batch_size==1时把这个维度去掉，其它代码直接去[iBatch]就可以了
                            # 用find_peaks求output的峰值索引
                            idx, _ = find_peaks(output.cpu().numpy(), distance=3/self.BPMresol)
                            # 对峰值索引对应的output值进行降序排序
                            highestPeak = torch.argsort(-output[idx]).cpu()
                            # 获得最高的Np个峰值索引，从而得到呼吸率估计值，转换成tensor
                            pred_val = torch.from_numpy(idx[highestPeak][:cfg['Np']] * self.BPMresol)

                            # pred_val = (torch.argsort(-output)[:cfg['Np']] * self.BPMresol).cpu()
                            # 对呼吸率估计值进行升序排序
                            pred_val = torch.sort(pred_val)[0]
                            print("pred_val:", pred_val)
                            print("cfg['gt']:", cfg['gt'])
                            rmse = torch.sqrt(torch.mean((pred_val[:cfg['Np']] - cfg['gt'][:cfg['Np']]) ** 2))
                            print('Epoch: [{0}][{1}/{2}]\t'
                                'Loss {loss:.4f}\t, RMSE {rmse:.4f}'.format(
                                epoch, i, len_train_loader, loss=loss.item(), rmse=rmse.item()))
                        else:
                            se = torch.zeros(1)
                            for iBatch in range(len(output)):
                                # print("output.shape:", output.shape)
                                # print("cfg['Np'].shape:", cfg['Np'].shape)
                                # print("cfg['Np']:", cfg['Np'])
                                # 用find_peaks求output[iBatch]的峰值索引
                                idx, _ = find_peaks(output[iBatch].cpu().numpy(), distance=3/self.BPMresol)
                                # 对峰值索引对应的output[iBatch]值进行降序排序
                                highestPeak = torch.argsort(-output[iBatch][idx]).cpu()
                                # 获得最高的Np个峰值索引，从而得到呼吸率估计值，转换成tensor
                                pred_val = torch.from_numpy(idx[highestPeak][:cfg['Np'][iBatch]] * self.BPMresol)

                                # pred_val = (torch.argsort(-output[iBatch])[:cfg['Np'][iBatch]] * self.BPMresol).cpu()
                                # 对呼吸率估计值进行升序排序
                                pred_val = torch.sort(pred_val)[0]
                                if i % 10 == 1 and iBatch % 4 == 0:
                                    # print("pred_val.shape:", pred_val.shape)
                                    print("pred_val, cfg['gt'][iBatch]:", pred_val, cfg['gt'][iBatch])
                                se += torch.sum((pred_val[:cfg['Np'][iBatch]] - cfg['gt'][iBatch][:cfg['Np'][iBatch]]) ** 2)
                                if cfg['na'][iBatch] == '001':
                                    se_001 += torch.sum((pred_val[:cfg['Np'][iBatch]] - cfg['gt'][iBatch][:cfg['Np'][iBatch]]) ** 2)
                            se_total += se
                            Np_total += torch.sum(cfg['Np'])
                            loss_total += loss.item()
                            rmse = torch.sqrt(se / torch.sum(cfg['Np']))
                            print('Epoch: [{0}][{1}/{2}]\t'
                                'Loss {loss:.4f}\t'
                                'RMSE {rmse:.4f}\t'.format(
                                epoch, i, len_train_loader, loss=loss.item(), rmse=rmse.item()))
            rmse_total = torch.sqrt(se_total / Np_total)
            rmse_001 = torch.sqrt(se_001 / 74)
            loss_total = loss_total / len_train_loader
            # print("loss_total, rmse_total:", loss_total, rmse_total)
            print("loss_total, rmse_total, rmse_001:", loss_total, rmse_total, rmse_001)
            writer.add_scalar('train_loss', loss_total, epoch)
            writer.add_scalar('train_rmse', rmse_total, epoch)
            writer.add_scalar('train_rmse_001', rmse_001, epoch)
            self.best_loss = loss_total
            self.best_rmse = rmse_total
            self.model.eval()

        writer.close()
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

    def model_predict(self, Ridx):
        self.test_loader = load_data_from_txt(
                                            Ridx = Ridx, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
                                            no_sample=self.no_sample, # 设置读取哪种txt文件，90样本或者320样本
                                            BPMresolution = self.BPMresol, # 设置BPM分辨率
                                            batch_size = 1, # 设置batch大小
                                            shuffle = False, # 设置是否打乱数据
                                            num_workers = 2, # 设置读取数据的线程数量
                                    )
        self.model.eval()
        with torch.no_grad():
            pred_val_file = [] # 每个文件的预测值列表
            na_last = ['']  # Fix for possibly unbound variable
            for _, (_, cfg) in enumerate(self.test_loader):
                na_last = cfg['na']
                break
            print("Prediciting file: ", na_last)
            for _, (x_in, cfg) in enumerate(self.test_loader):  # Fix for unused variable
                x_in = x_in.cuda()
                x_in = torch.unsqueeze(x_in, 1) # [batch=1, 1, 320, 600]
                output = self.model(x_in).squeeze()
                # pred_val = (torch.argsort(-output)[:, :cfg['Np']] * self.BPMresol).cpu()
                # pred_val = torch.sort(pred_val, dim=1)[0]
                # pred_val = pred_val.squeeze().numpy()
                # 用find_peaks求output[iBatch]的峰值索引
                idx, _ = find_peaks(output.cpu().numpy(), distance=3/self.BPMresol)
                # 对峰值索引对应的output[iBatch]值进行降序排序
                highestPeak = torch.argsort(-output[idx]).cpu()
                # 获得最高的Np个峰值索引，从而得到呼吸率估计值，转换成tensor
                pred_val = torch.from_numpy(idx[highestPeak][:cfg['Np']] * self.BPMresol)

                # pred_val = (torch.argsort(-output[iBatch])[:cfg['Np'][iBatch]] * self.BPMresol).cpu()
                # 对呼吸率估计值进行升序排序
                pred_val = torch.sort(pred_val)[0]
                if cfg['na'] != na_last:
                    print("Prediciting file: ", cfg['na'])
                    save_data_to_txt(pred_val_file, na_last, Ridx)
                    pred_val_file = []
                    na_last = cfg['na']
                pred_val_file.append(pred_val)
            save_data_to_txt(pred_val_file, cfg['na'], Ridx)



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
