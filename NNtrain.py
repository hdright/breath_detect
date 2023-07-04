import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import torch.nn as nn
import random
from bdn.data_old import load_data
from bdn.csi_bd import CNN_trainer, LSTM_trainer

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


SEED = 42
seed_everything(SEED)


def train_LSTM():
    print("="*30)
    print("RegLSTM")
    # print("compressed codeword bits: {}".format(bits))
    agent3 = LSTM_trainer(epochs=40, net="RegLSTM")
    x3, agent3_loss, t3 = agent3.model_train()
    print("RegLSTM")
    print(agent3_loss)
    print("average time used is:", t3)
    plt.plot(x3, agent3_loss, label="RegLSTM")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs epochs")
    plt.savefig("MSE vs epochs.png")


def train_CNN():
    print("="*30)
    # print("compressed codeword bits: {}".format(bits))
    train_now = True # TODO
    # train_now = False
    no_sample = 960  # TODO 180对应3x30场景，640对应4x80场景
    net = "BDInception3" # TODO
    # net = "BDCNN"
    print("net: ", net)
    if net == "BDCNN":
        lr = 1e-4
        dataBorrow = False
    else:
        lr = 1e-3
        dataBorrow = True
    if no_sample == 90:
        batch_size = 34
        Np2extend = []  # [2, 3]
        preProcList = ['amp']
        bnr_range = [6, 45]
    elif no_sample == 180:
        # batch_size = 45
        batch_size = 34
        # batch_size = 44 # dataBorrow=True
        Np2extend = []  # [1, 2, 3]
        # 1\2种学习的数据分别用什么['amp', 'diffPha', 'ampRatio', 'pha']
        # preProcList = ['amp', 'ampRatio']
        preProcList = ['amp', 'diffPha']
        bnr_range = [6, 45]
    elif no_sample == 960:
        batch_size = 32
        Np2extend = []
        # preProcList = ['amp', 'diffPha', 'diffSani']
        preProcList = ['amp', 'diffPha', 'ampRaBnr']
        preProcList = ['amp', 'diffPha', 'ampRatio']
        bnr_range = [6, 45]
    elif no_sample == 270:
        batch_size = 34
        Np2extend = []
        preProcList = ['amp', 'diffPha', 'diffSani']
        bnr_range = [6, 45]
    else:
        batch_size = 32  # 320长度的数据共有416种情况, 90长度的数据共有68种情况
        Np2extend = []
        bnr_range = [6, 45]
        # 1\2种学习的数据分别用什么['amp', 'diffPha', 'ampRatio', 'pha']
        preProcList = ['amp', 'diffPha']
    print("preProcList: ", preProcList)
    agent3 = CNN_trainer(epochs=72,# TODO
                        #   net="BDCNN",
                         net=net,
                         train_now=train_now,
                         no_sample=no_sample,
                         pre_sg=[5, 3],  # 数据预处理的savgol参数
                         dataBorrow=dataBorrow, # 3x30场景是否借用4x80场景数据进行训练
                         preProcList=preProcList,  # 1\2种学习的数据分别用什么['amp', 'diffPha', 'ampRatio', 'pha']
                         Np2extend=Np2extend,  # 对Np=？的数据进行扩展
                         bnr_range=bnr_range,  # bnr范围
                         aux_logits=False,
                         BPMresol=0.1,
                         breathEnd=1,
                         batch_size=batch_size,
                         learning_rate=lr,  # 学习率
                         lr_decay_freq=40,  # 多少个epoch衰减# TODO
                         lr_decay=0.1,
                         num_workers=0,
                         print_freq=1,
                         train_test_ratio=0.8)
    if train_now:
        agent3.model_train()
        agent3.model_save()
    else:
        if no_sample == 180:
            # agent3.model_load("breath_detect/model_save/2023-06-24_09-52-56-sc14p45-3x30-noStdAmp-indepStdPha/BDCNN_2023-06-24_09-52-56.pkl")
            # agent3.model_load("breath_detect/model_save/2023-06-28_21-09-39-180-inputSg53/BDCNN_2023-06-28_21-09-39.pkl")  # best
            # agent3.model_load(
            #     "breath_detect/model_save/BDInception3/2023-07-01_11-30-27-3x30-incep-sg53-ep160de82-lrm3/BDCNN_2023-07-01_11-30-27.pkl")
            agent3.model_load(
                "breath_detect/model_save/BDInception3/2023-07-01_21-08-58-3x30-incep-borrow-amp-diffPha-sg53-ep80de40-lrm3/BDCNN_2023-07-01_21-08-58.pkl")# 13.90
            # agent3.model_load(
            #     "breath_detect/model_save/BDInception3/2023-07-02_16-15-07-3x30-incep-borrow-amp-diffPha-sg87-ep80de40-lrm3/BDCNN_2023-07-02_16-15-07.pkl")
        elif no_sample == 270:
            agent3.model_load("breath_detect/model_save/BDCNN/b2023-07-03_15-20-38-3x30BDCNN-amp-diffPha-diffSani-sg53-ep160de80-lrm4/BDCNN_2023-07-03_15-20-38.pkl")#12.08
        elif no_sample == 640:
            # agent3.model_load("breath_detect/model_save/2023-06-23_14-20-22sc14p95-idepStdDiffPhase-4x80-ep160/BDCNN_2023-06-23_14-20-22.pkl")
            # agent3.model_load("breath_detect/model_save/2023-06-28_20-11-13-640-inputSg53/BDCNN_2023-06-28_20-11-13.pkl")  
            # agent3.model_load(
            #     "breath_detect/model_save/2023-06-30_11-57-04-datasg53-ampRatio-3x30-BN-ep160/BDCNN_2023-06-30_11-57-04.pkl")
            agent3.model_load("breath_detect/model_save/BDInception3/2023-06-30_19-29-58-4x80-incep-sg53-ep80de59-lrm3/BDCNN_2023-06-30_19-29-58.pkl")# best
        elif no_sample == 180640:
            agent3.model_load(
                "breath_detect/model_save/2023-06-28_18-24-59-180640-90extend/BDCNN_2023-06-28_18-24-59.pkl")
    agent3.model_predict(Ridx=3)
    # print("BDCNN")
    # print(agent3_loss)
    # print("average time used is:", t3)
    # plt.plot(x3, agent3_loss, label="RegLSTM")
    # plt.legend()
    # plt.xlabel("epochs")
    # plt.ylabel("MSE")
    # plt.title("MSE vs epochs")
    # plt.savefig("MSE vs epochs.png")


if __name__ == '__main__':
    train_CNN()
