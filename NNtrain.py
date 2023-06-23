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
    print("BDCNN")
    # print("compressed codeword bits: {}".format(bits))
    # train_now = False
    train_now = True
    no_sample = 640
    if no_sample == 90:
        batch_size = 34
    elif no_sample == 180:
        # batch_size = 45
        batch_size = 34
    else:
        batch_size = 32 # 320长度的数据共有416种情况, 90长度的数据共有68种情况
    agent3 = CNN_trainer(epochs=160,
                            net="BDCNN",
                         train_now=train_now,
                         no_sample=no_sample,
                         BPMresol=0.1,
                         breathEnd=1,
                         batch_size=batch_size, 
                         learning_rate=1e-4,
                         lr_decay_freq=160,
                         lr_decay=0.1,
                         num_workers=0,
                         print_freq=1,
                         train_test_ratio=0.8)
    if train_now:
        agent3.model_train()
        agent3.model_save()
    else:
        agent3.model_load("2023-06-22_13-44-31-BDCNN640-ep320decay160/BDCNN_2023-06-22_13-44-31.pkl")
    agent3.model_predict(Ridx=2)
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