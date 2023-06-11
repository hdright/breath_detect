import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
import torch.nn as nn
import random
from MoslehLSTM.data import load_data
from MoslehLSTM.csi_bd import model_trainer

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

def train_all(bits):
    print("="*30)
    print("RegLSTM")
    # print("compressed codeword bits: {}".format(bits))
    agent3 = model_trainer(epochs=40, net="RegLSTM")
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


# train_all(256)
# train_all(128)
# train_all(64)
train_all(32)
