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
from scipy.signal import find_peaks, savgol_filter

# from bdn.loss import NMSE_cuda, NMSELoss, CosSimilarity, rho
from bdn.backbone import RegLSTM, BDCNN, BDCNNold, BDCNN_ND, BDInception3
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


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def create_dir(directory): # 创建（尚未存在的）空目录函数
    try:
        os.mkdir(directory)
    except FileNotFoundError:
        os.makedirs(directory)
    except FileExistsError:
        pass 


SEED = 42
print("seeding everything...")
seed_everything(SEED)
print("initializing parameters...")


class CNN_trainer():

    def __init__(self,
                 epochs,
                 net,
                 #  feedbackbits=128,
                 train_now=True,
                 no_sample=320,  # 设置读取哪种txt文件，90样本或者320样本
                 preProcList=['amp', 'diffPha'], # 1\2种学习的数据分别用什么['amp', 'diffPha', 'ampRatio', 'pha']
                 pre_sg = [8, 7], # 数据预处理savgol滤波器参数
                 dataBorrow=False, # 3x30场景是否借用4x80场景数据进行训练
                 bnr_range=[6, 45], # bnrBPM的范围
                 Np2extend=[2, 3], # 训练时扩展几个人的场景的数据集
                 aux_logits=True, # 是否使用辅助分类器
                 BPMresol=1.0,
                 breathEnd=1,
                 batch_size=1,
                 learning_rate=1e-3,
                 lr_decay_freq=30,
                 lr_decay=0.1,
                 #  best_loss=100,
                 num_workers=0,
                 print_freq=25,
                 train_test_ratio=0.8):
        self.net = net
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
        self.model_path = os.path.join(model_path, self.net, self.model_time)
        if train_now:
            create_dir(self.model_path)

        self.no_sample = no_sample  # 读取的样本数
        self.dataBorrow = dataBorrow  # 3x30场景是否借用4x80场景数据进行训练
        self.preProcList = preProcList  # 1\2种学习的数据分别用什么
        self.pre_sg = pre_sg  # 数据预处理savgol滤波器参数
        self.Np2extend = Np2extend  # 扩展几个人的场景
        self.aux_logits = aux_logits and (net == 'BDInception3')  # 是否使用辅助分类器
        # 要估计的呼吸频率范围
        self.BPMresol = BPMresol
        self.breadthEnd = breathEnd
        # resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
        self.bpmMinMax = [5, 50]  # 呼吸频率范围
        self.bnr_range = bnr_range  # bnrBPM的范围
        bpmRange = np.arange(0, 60, BPMresol)
        noBpmPoints = len(bpmRange)  # 要估计的呼吸频率个数

        if self.aux_logits:
            self.model = eval(net)(input_sample=no_sample, output_size=noBpmPoints, aux_logits=True)
        else:
            self.model = eval(net)(input_sample=no_sample, output_size=noBpmPoints)
        print("noBpmPoints: ", noBpmPoints)
        self.x_label = []
        self.y_label = []
        self.ys_label = []
        self.t_label = []

        if len(gpu_list.split(',')) > 1:
            self.model = torch.nn.DataParallel(
                self.model).cuda()  # model.module
        else:
            self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_test = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

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
            # train002009 = './chusai_data/TestData/train_shuffle_loader_stdampfft_stdamp_gausssig100.pkl'
            # train002009 = './chusai_data/TestData/train_shuffle_loader_StdAmpFft_indepstdamp_gausssig100.pkl'
            train002009 = './chusai_data/TestData/train_shuffle_loader_colStdAmpFft_stdAmp_gausssig100.pkl'
            # train002009 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1000.pkl'
            # train002009 = './chusai_data/TestData/train_shuffle_loader_hampel_stdfft_gaussianlabelsig100.pkl'
            # train001 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90.pkl'
            # train001 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90_more_reasonable_fftstretch.pkl'
            # train001 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90_nostretch.pkl'
            train001 = './chusai_data/TestData/train_shuffle_loader_stdampfft_stdamp_gaussianlabelsig1_90_nostretch.pkl'
            # train001_180 = './chusai_data/TestData/train_shuffle_loader_stdampfft_stdamp_gaussianlabelsig1_180_nostretch.pkl'
            # train001_180 = './chusai_data/TestData/train_shuffle_loader_stdampfft_stdamp_gaussianlabelsig1_180ronly001_nostretch.pkl'
            # train001_180 = './chusai_data/TestData/train_shuffle_stdampfft_stdamp_indepStdDiffPha_gaussianlabelsig1_180ronly001_nostretch.pkl'
            # train001_180 = './chusai_data/TestData/train_shuffle_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180only001_nostretch.pkl'
            # train001_180 = './chusai_data/TestData/train_shuffle_sg53_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180only001_nostretch.pkl'  # best 2nd
            train001_270 = './chusai_data/TestData/train_270_sg53_stdampfft_nostdamp_indepStdDiffPhaAndDiffSani_gaussianlabelsig100_180only001_nostretch.pkl'  
            # train001_180 = './chusai_data/TestData/train_shuffle_sg53_stdampfft_nostdamp_indepStdAmpRa_gaussianlabelsig100_180only001_nostretch.pkl'  # 
            train001_180_borrow = './chusai_data/TestData/train_shuffle_borrow_sg53_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180-3to6-5to35_nostretch.pkl' # best
            train001_270_borrow = './chusai_data/TestData/train_shuffle_borrow_sg53_stdampfft_nostdamp_indepStdDiffPhaAndDiffSani_gaussianlabelsig100_180-3to6-5to35_nostretch.pkl' # 
            # train001_180_borrow = './chusai_data/TestData/train_shuffle_borrow_sg87_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180-3to6-5to35_nostretch.pkl' # 
            # train001_180_borrow = './chusai_data/TestData/train_shuffle_borrow_sg53_stdampfft_uniStdAmp_indepStdDiffPha_gaussianlabelsig100_180-3to6-5to35_nostretch.pkl' #
            # train001_180_borrow = './chusai_data/TestData/train_shuffle_borrow-batch44_sg53_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180-3to6-5to35_nostretch.pkl' # 
            # train001_180_borrow = './chusai_data/TestData/train_shuffle_borrow_sg53_stdampfft_nostdamp_indepStdAmpRa_gaussianlabelsig100_180-3to6-5to35_nostretch.pkl' 
            # train001_180 = './chusai_data/TestData/train_shuffle_sg53_stdampfft_indepStdamp_indepStdAmpRa_gaussianlabelsig100_180only001_nostretch.pkl'  # 
            # train001_180 = './chusai_data/TestData/train_shuffle_sg53_stdAmpFft_indepStdAmpRatio_indepStdDiffPha_gaussianlabelsig100_180only001_nostretch.pkl'  # 
            # train001_180 = './chusai_data/TestData/train_shuffle_sg87_stdAmpFft_indepStdAmpRatio_indepStdDiffPha_gaussianlabelsig100_180only001_nostretch.pkl'  # 
            # train001_180_extend = './chusai_data/TestData/train_easyExtend_shuffle_sg53_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180only001_nostretch.pkl'  # 
            train001_180_extend = './chusai_data/TestData/train_easyExtend123_shuffle_sg53_stdampfft_nostdamp_indepStdDiffPha_gaussianlabelsig100_180only001_nostretch.pkl'  # 
            # train90320 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90320.pkl'
            train90320 = './chusai_data/TestData/train_shuffle_loader_stdfft_gaussianlabelsig1_90320_more_reasonable_fftstretch.pkl'
            # train002009_640 = './chusai_data/TestData/train_shuffle_640_colStdAmpFft_stdAmp_gausssig100.pkl'
            # train002009_640 = './chusai_data/TestData/train_shuffle_640_colStdAmpFft_stdAmp_deltaphase_gausssig100.pkl'
            # train002009_640 = './chusai_data/TestData/train_shuffle_640_colStdAmpFft_stdAmp_diffphase_gausssig100.pkl'
            # train002009_640 = './chusai_data/TestData/train_shuffle_640_colStdAmpFft_stdAmp_indepStdDiffPhase_gausssig100.pkl'
            train002009_640 = './chusai_data/TestData/train_shuffle_640_sg53_colStdAmpFft_stdAmp_indepStdDiffPhase_gausssig100.pkl'  # best 2nd
            # train002009_640 = './chusai_data/TestData/train_shuffle_640_sg53_colStdAmpFft_indepStdAmpRatio_indepStdDiffPhase_gausssig100.pkl'  # 
            # train002009_640 = './chusai_data/TestData/train_shuffle_640_colStdAmpFft_stdAmp_indepStdDiffPhase_gausssig25.pkl' # bad
            # train002009_960 = './chusai_data/TestData/train_shuffle_960_sg53_colStdAmpFft_stdAmp_indepStdDiffPhaseAndDiffSani_gausssig100.pkl'  # 
            train002009_960 = './chusai_data/TestData/train_shuffle_960_sg53_colStdAmpFft_stdAmp_indepStdDiffPhase_ampRaBnr_gausssig100.pkl'  
            # train002009_960 = './chusai_data/TestData/train_shuffle_960_sg53_colStdAmpFft_stdAmp_indepStdDiffPhase_ampRa_gausssig100.pkl'  # best
            # train002005_960 = './chusai_data/TestData/train_shuffle_002005960_sg53_colStdAmpFft_stdAmp_indepStdDiffPhase_ampRa_gausssig100.pkl'  #
            train180640 = './chusai_data/TestData/train_shuffle_180noStdAmp_640stdAmp_indepStdDiffPhase_gausssig100.pkl'  # very bad
            train_pkl = train002009_960
            if os.path.exists(train_pkl):
                print('Loading train_shuffle_loader...')
                with open(train_pkl, 'rb') as f:
                    self.train_shuffle_loader = pickle.load(f)
            else:
                self.train_shuffle_loader = load_data_from_txt(
                    # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
                    # '''test loader 也要改''' TODO
                    Ridx=0,
                    no_sample=self.no_sample,  # 设置读取哪种txt文件，90样本或者320样本
                    dataBorrow=self.dataBorrow,  # 3x30场景是否借用4x80场景数据进行训练
                    preProcList=self.preProcList,  # 1\2种学习的数据分别用什么
                    pre_sg=self.pre_sg,  # 预处理savgol滤波器参数
                    Np2extend=self.Np2extend,  # 设置是否对90样本进行扩展
                    BPMresolution=self.BPMresol,  # 设置BPM分辨率
                    bnr_range=self.bnr_range,  # 设置BPM范围
                    batch_size=self.batch_size,  # 设置batch大小
                    shuffle=True,  # 设置是否打乱数据
                    num_workers=2,  # 设置读取数据的线程数量
                )
                with open(train_pkl, 'wb') as f:
                    pickle.dump(self.train_shuffle_loader, f)

    def model_save(self, name="BDCNN"):#, path="./model_save"):
        print('Saving model...')
        # path = os.path.join(path, self.net, self.model_time)
        # create_dir(path)
        model_name = name + '_' + self.model_time + '.pkl'
        modelPATH = os.path.join(self.model_path, model_name)
        torch.save({'state_dict': self.model.state_dict(), }, modelPATH)
        # 保存模型结构
        with open(os.path.join(self.model_path, model_name + '.txt'), 'w') as f:
            print("epochs: ", self.epochs, file=f)
            print("lr: ", self.learning_rate, file=f)
            print("lr_decay_freq: ", self.lr_decay_freq, file=f)
            print("lr_decay: ", self.lr_decay, file=f)
            print("batch_size: ", self.batch_size, file=f)
            print("best_loss: ", self.best_loss, file=f)
            print("best_rmse: ", self.best_rmse, file=f)
            print("no_sample: ", self.no_sample, file=f)
            print("BPMresol: ", self.BPMresol, file=f)
            print("bnr_range: ", self.bnr_range, file=f)
            print("dataBorrow: ", self.dataBorrow, file=f)
            print("preProcList: ", self.preProcList, file=f)
            print("pre_sg: ", self.pre_sg, file=f)
            print("Np2extend: ", self.Np2extend, file=f)
            print("aux_logits: ", self.aux_logits, file=f)
            print(self.model, file=f)
        print('Model saved!')

    def model_load(self, name, path="../"):
        print('Loading model...')
        modelPATH = os.path.join(path, name)
        model_dict = torch.load(modelPATH)
        self.model.load_state_dict(
            model_dict['state_dict'], strict=False)  # type: ignore

    def model_train(self):
        # logdir = os.path.join('./model_save/', self.net, self.model_time)
        # create_dir(logdir)
        writer = SummaryWriter(log_dir=self.model_path)
        for epoch in range(self.epochs):
            print('========================')
            print('lr:%.4e' % self.optimizer.param_groups[0]['lr'])
            # train model
            self.model.train()

            # decay lr
            # if epoch % self.lr_decay_freq == 0 and epoch > 0:
            if epoch == self.lr_decay_freq:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.lr_decay

            # training...
            se_total = torch.zeros(1)
            se_001 = torch.zeros(1)
            Np_total = torch.zeros(1)
            Np_001 = torch.zeros(1)
            loss_total = torch.zeros(1)
            len_train_loader = len(self.train_shuffle_loader)
            for i, (x_in, label, cfg) in enumerate(self.train_shuffle_loader):
                self.optimizer.zero_grad()
                x_in = x_in.cuda()  # input [batch=1, 320, 600]
                # x_in = torch.unsqueeze(x_in, 1) # [batch=1, 1, 320, 600] # 取消这个操作，因为在load_data_from_txt中已经增加了一个维度
                # print("x_in.shape:", x_in.shape)
                if self.aux_logits:
                    output, aux_logits = self.model(x_in)
                    label = label.cuda()
                    loss0 = self.criterion(output, label)
                    loss1 = self.criterion(aux_logits, label)
                    loss = loss0 + 0.4 * loss1
                else:
                    output = self.model(x_in)  # .squeeze()
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
                            # batch_size==1时把这个维度去掉，其它代码直接去[iBatch]就可以了
                            output = output.squeeze()
                            # 用find_peaks求output的峰值索引
                            idx, _ = find_peaks(
                                output.cpu().numpy(), distance=3/self.BPMresol)
                            # 对峰值索引对应的output值进行降序排序
                            highestPeak = torch.argsort(-output[idx]).cpu()
                            # 获得最高的Np个峰值索引，从而得到呼吸率估计值，转换成tensor
                            pred_val = torch.from_numpy(
                                idx[highestPeak][:cfg['Np']] * self.BPMresol)

                            # pred_val = (torch.argsort(-output)[:cfg['Np']] * self.BPMresol).cpu()
                            # 对呼吸率估计值进行升序排序
                            pred_val = torch.sort(pred_val)[0]
                            print("pred_val:", pred_val)
                            print("cfg['gt']:", cfg['gt'])
                            rmse = torch.sqrt(torch.mean(
                                (pred_val[:cfg['Np']] - cfg['gt'][:cfg['Np']]) ** 2))
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
                                if self.no_sample % 90 == 0:
                                    output_sg = savgol_filter(
                                        output[iBatch].cpu().numpy(), 5, 3)
                                    idx, _ = find_peaks(
                                        output_sg, distance=6/self.BPMresol)  # 3x30场景，复赛间隔6
                                else:
                                    output_sg = output[iBatch].cpu().numpy()
                                    idx, _ = find_peaks(
                                        output_sg, distance=3/self.BPMresol)
                                # 排除超出self.bpmMinMax范围的峰值索引
                                idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                                    idx * self.BPMresol < self.bpmMinMax[1])]
                                # 对峰值索引对应的output[iBatch]值进行降序排序
                                highestPeak = torch.argsort(
                                    -output[iBatch][idx]).cpu()
                                # 获得最高的Np个峰值索引，从而得到呼吸率估计值，转换成tensor
                                pred_val = torch.from_numpy(
                                    idx[highestPeak][:cfg['Np'][iBatch]] * self.BPMresol)

                                # pred_val = (torch.argsort(-output[iBatch])[:cfg['Np'][iBatch]] * self.BPMresol).cpu()
                                # 对呼吸率估计值进行升序排序
                                pred_val = torch.sort(pred_val)[0]
                                if i % 10 == 1 and iBatch % 4 == 0:
                                    # print("pred_val.shape:", pred_val.shape)
                                    print(
                                        "pred_val, cfg['gt'][iBatch]:", pred_val, cfg['gt'][iBatch])
                                se += torch.sum(
                                    (pred_val[:cfg['Np'][iBatch]] - cfg['gt'][iBatch][:cfg['Np'][iBatch]]) ** 2)
                                if cfg['na'][iBatch] == '001':
                                    se_001 += torch.sum(
                                        (pred_val[:cfg['Np'][iBatch]] - cfg['gt'][iBatch][:cfg['Np'][iBatch]]) ** 2)
                                    Np_001 += torch.sum(cfg['Np'][iBatch])
                            se_total += se
                            Np_total += torch.sum(cfg['Np'])
                            loss_total += loss.item()
                            rmse = torch.sqrt(se / torch.sum(cfg['Np']))
                            print('Epoch: [{0}][{1}/{2}]\t'
                                  'Loss {loss:.4f}\t'
                                  'RMSE {rmse:.4f}\t'.format(
                                      epoch, i, len_train_loader, loss=loss.item(), rmse=rmse.item()))
            rmse_total = torch.sqrt(se_total / Np_total)
            rmse_001 = torch.sqrt(se_001 / Np_001)
            loss_total = loss_total / len_train_loader
            # print("loss_total, rmse_total:", loss_total, rmse_total)
            print("loss_total, rmse_total, rmse_001:",
                  loss_total, rmse_total, rmse_001)
            writer.add_scalar('train_loss', loss_total, epoch)
            writer.add_scalar('train_rmse', rmse_total, epoch)
            writer.add_scalar('train_rmse_001', rmse_001, epoch)
            self.best_loss = loss_total
            self.best_rmse = rmse_total
            self.model.eval()

        writer.close()

    def model_predict(self, Ridx):
        self.test_loader = load_data_from_txt(
            # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
            Ridx=Ridx,
            no_sample=self.no_sample,  # 设置读取哪种txt文件，90样本或者320样本
            preProcList=self.preProcList,  # 1\2种学习的数据分别用什么['amp', 'diffPha', 'ampRatio', 'pha']
            bnr_range=self.bnr_range,  # 设置bnr读取哪个范围的数据
            BPMresolution=self.BPMresol,  # 设置BPM分辨率
            pre_sg=self.pre_sg,  # 预处理savgol滤波器参数
            batch_size=1,  # 设置batch大小
            shuffle=False,  # 设置是否打乱数据
            num_workers=2,  # 设置读取数据的线程数量
        )
        self.model.eval()
        if self.net == "BDCNN": # TODO
            self.model.apply(apply_dropout)  # eval时依然使用dropout
        with torch.no_grad():
            pred_val_file = []  # 每个文件的预测值列表
            na_last = ['']  # Fix for possibly unbound variable
            for _, (_, cfg) in enumerate(self.test_loader):
                na_last = cfg['na']
                break
            print("Prediciting file: ", na_last)
            pic_folder_time = time.strftime(
                        "%Y-%m-%d_%H:%M:%S", time.localtime())
            pic_folder = "./pic/" + pic_folder_time
            create_dir(pic_folder)
            for i_test, (x_in, cfg) in enumerate(self.test_loader):  # Fix for unused variable
                x_in = x_in.cuda()
                # x_in = torch.unsqueeze(x_in, 1) # [batch=1, 1, 320, 600]
                print("=====================================")
                if self.net == "BDCNN" or self.net == "BDInception3":
                    if self.net == "BDCNN":
                        avg_time = 100 # TODO(best 100)
                    else:
                        avg_time = 1
                    pred_val_list = []
                    # 获取当前字符串
                    pic_time = str(i_test)
                    for _ in range(avg_time):
                        output = self.model(x_in).squeeze()
                        if self.no_sample % 90 == 0 or self.no_sample % 320 == 0:
                            output_sg = savgol_filter(
                                output.cpu().numpy(), 5, 3)
                            # output_sg = savgol_filter(
                            #     output.cpu().numpy(), 21, 8) # 和5,3没区别？
                            # output_sg = savgol_filter(output.cpu().numpy(), 8, 7) #不如5,3
                            # output_sg = output.cpu().numpy()
                            if self.no_sample % 90 == 0:
                                if cfg['Np'] == 2:
                                    idx, _ = find_peaks(
                                        output_sg, distance=8/self.BPMresol)  # 3x30场景，间隔8
                                elif cfg['Np'] == 3:
                                    idx, _ = find_peaks(
                                        output_sg, distance=5/self.BPMresol)
                                else:
                                    idx, _ = find_peaks(
                                        output_sg, distance=3/self.BPMresol)
                                # idx, _ = find_peaks(
                                #     output_sg, distance=3/self.BPMresol) # TODO 
                                # 排除超出self.bpmMinMax范围的峰值索引
                                if cfg['Np'] == 1:
                                    idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                                        idx * self.BPMresol < self.bpmMinMax[1])]
                                elif cfg['Np'] == 2:
                                    idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                                        idx * self.BPMresol < 30)]
                                elif cfg['Np'] == 3:
                                    idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                                        idx * self.BPMresol < 30)]
                            elif self.no_sample % 320 == 0:
                                idx, _ = find_peaks(
                                    output_sg, distance=3/self.BPMresol)
                                idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                                        idx * self.BPMresol < self.bpmMinMax[1])]
                            else:
                                idx = np.zeros(1)
                            
                            picNp3 = os.path.join(pic_folder, "%s_Np3.jpg" % pic_time)
                            picNp2 = os.path.join(pic_folder, "%s_Np2.jpg" % pic_time)
                            if not os.path.exists(picNp3) and cfg['Np'] == 3:
                                plt.figure()
                                plt.plot(output_sg)
                                plt.plot(idx, output_sg[idx], 'x')
                                plt.savefig(picNp3)
                            if not os.path.exists(picNp2) and cfg['Np'] == 2:
                                plt.figure()
                                plt.plot(output_sg)
                                plt.plot(idx, output_sg[idx], 'x')
                                plt.savefig(picNp2)
                        else: # 90样本/320样本
                            output_sg = output.cpu().numpy()
                            idx, _ = find_peaks(
                                output_sg, distance=3/self.BPMresol)
                            # 排除超出self.bpmMinMax范围的峰值索引
                            idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                                idx * self.BPMresol < self.bpmMinMax[1])]
                        highestPeak = torch.argsort(-output[idx]).cpu()
                        # pred_val = torch.from_numpy(idx[highestPeak][:cfg['Np']] * self.BPMresol)
                        # pred_val = torch.sort(pred_val)[0]
                        pred_val = idx[highestPeak][:cfg['Np']] * self.BPMresol
                        pred_val = np.sort(pred_val)
                        pred_val_list.append(pred_val)
                    if avg_time > 1:
                        # 控制z-score不会导致性能更好# Calculate the z-score of each array
                        z_scores = np.abs(
                            (pred_val_list - np.mean(pred_val_list, axis=0)) / np.std(pred_val_list))
                        if cfg['Np'] == 3:
                            for i in range(len(pred_val_list)):
                                print("pred_val_list, z_scores:",
                                      pred_val_list[i], z_scores[i])
                            # print("pred_val_list:", pred_val_list)
                            # print("z_scores:", z_scores)
                        # Remove arrays with a z-score greater than 3
                        pred_val_list = [pred_val_list[i] for i in range(
                            len(pred_val_list)) if np.max(z_scores[i]) < 1]
                        pred_val = np.mean(pred_val_list, axis=0)
                else:  # 无dropout，不取平均
                    output = self.model(x_in).squeeze()
                    # pred_val = (torch.argsort(-output)[:, :cfg['Np']] * self.BPMresol).cpu()
                    # pred_val = torch.sort(pred_val, dim=1)[0]
                    # pred_val = pred_val.squeeze().numpy()
                    # 用find_peaks求output[iBatch]的峰值索引
                    idx, _ = find_peaks(
                        output.cpu().numpy(), distance=3/self.BPMresol)
                    # 排除超出self.bpmMinMax范围的峰值索引
                    idx = idx[(idx * self.BPMresol > self.bpmMinMax[0]) & (
                        idx * self.BPMresol < self.bpmMinMax[1])]
                    # 对峰值索引对应的output[iBatch]值进行降序排序
                    highestPeak = torch.argsort(-output[idx]).cpu()
                    # 获得最高的Np个峰值索引，从而得到呼吸率估计值，转换成tensor
                    pred_val = torch.from_numpy(
                        idx[highestPeak][:cfg['Np']] * self.BPMresol)
                    pred_val = torch.sort(pred_val)[0]

                # pred_val = (torch.argsort(-output[iBatch])[:cfg['Np'][iBatch]] * self.BPMresol).cpu()
                # 对呼吸率估计值进行升序排序

                if cfg['na'] != na_last:
                    print("Prediciting file: ", cfg['na'])
                    save_data_to_txt(pred_val_file, na_last, Ridx)
                    pred_val_file = []
                    na_last = cfg['na']
                if self.no_sample % 90 == 0:
                    if cfg['Np'] == 3:
                        pred_val[2] += 2
                #     elif cfg['Np'] == 2:
                #         pred_val[1] += 3
                pred_val_file.append(pred_val)
            save_data_to_txt(pred_val_file, cfg['na'], Ridx)


class LSTM_trainer():

    def __init__(self,
                 epochs,
                 net,
                 #  feedbackbits=128,
                 inp_dim=2,  # regLSTM
                 out_dim=1,  # regLSTM
                 mid_dim=40,  # regLSTM
                 linear_dim=250,  # regLSTM
                 mid_layers=2,  # regLSTM
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
            self.model = torch.nn.DataParallel(
                self.model).cuda()  # model.module
        else:
            self.model = self.model.cuda()

        # self.criterion = NMSELoss(reduction='mean')
        self.criterion = nn.MSELoss()
        # self.criterion_test = NMSELoss(reduction='sum')
        self.criterion_test = nn.MSELoss()
        # self.criterion_rho = CosSimilarity(reduction='mean')
        # self.criterion_test_rho = CosSimilarity(reduction='sum')
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

        # train_loader, test_loader, train_dataset, test_dataset, \
        # train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

        self.train_loader, self.test_loader, self.train_dataset,        self.test_dataset, self.train_shuffle_loader, self.test_shuffle_loader,        self.train_shuffle_dataset, self.test_shuffle_dataset = \
            load_data('./train_data', shuffle=True)

    def model_save(self, encodername, decodername):
        print('Saving model...')
        encoderPATH = os.path.join(model_path, encodername)
        decoderPATH = os.path.join(model_path, decodername)
        try:
            torch.save(
                {'state_dict': self.model.encoder.state_dict(), }, encoderPATH)
        except:
            torch.save(
                {'state_dict': self.model.module.encoder.state_dict(), }, encoderPATH)

        try:
            torch.save(
                {'state_dict': self.model.decoder.state_dict(), }, decoderPATH)
        except:
            torch.save(
                {'state_dict': self.model.module.decoder.state_dict(), }, decoderPATH)
#         print('Model saved!')
        self.best_loss = self.average_loss

    def model_load(self, encodername, decodername):
        encoderPATH = os.path.join(model_path, encodername)
        decoderPATH = os.path.join(model_path, decodername)

        encoder_dict = torch.load(encoderPATH)
        self.model.encoder.load_state_dict(
            encoder_dict['state_dict'], strict=False)  # type: ignore

        decoder_dict = torch.load(decoderPATH)
        self.model.decoder.load_state_dict(
            decoder_dict['state_dict'], strict=False)  # type: ignore

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
                    self.total_loss += self.criterion_test(
                        output, label).item()
                    # self.total_rho += self.criterion_rho(output,input).item()
                    # print(rho(output,input), type(rho(output,input)))
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

        # , self.ys_label
        return self.x_label, self.y_label, sum(self.t_label)/len(self.t_label)
