#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for loading data and pre-processing data
import numpy as np
import h5py
import torch
from chusai import *
from scipy.signal import savgol_filter
from torch.utils.data import Dataset

img_height = 16
img_width = 32
img_channels = 2

class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index] 

def load_data(
        file_path,
        shuffle = False,
        train_test_ratio=0.8,
        batch_size=30,
        num_workers=0,
        pin_memory=True,
        drop_last=True):

    print("loading data...")
    # mat = h5py.File(file_path + '/Hdata.mat', 'r')
    # data = np.transpose(mat['H_train'])
    # data = data.astype('float32')
    # data = np.reshape(data, [len(data), img_channels, img_height, img_width])
    data = np.load(file_path + '/CSI_train_conca.npy')
    dshape = data.shape
    data = data.astype('float32')
    label = np.load(file_path + '/GT_train.npy')
    label = label.astype('float32')
    # data_label = np.concatenate((data, label), axis=1)

    # if shuffle:
    #     data_copy = np.copy(data_label)
    #     np.random.shuffle(data_copy)
    #     data_shuffle = data_copy
        # data_transpose = data_copy.transpose()
        # np.random.shuffle(data_transpose)
        # data_shuffle = data_transpose.transpose()

    partition = int(data.shape[0] * train_test_ratio)
    x_train, x_test = data[:partition], data[partition:]
    y_train, y_test = label[:partition], label[partition:]
    # x_train_shuffle, x_test_shuffle = data_shuffle[:partition], data_shuffle[partition:]
    # 分离x_train, x_test的数据和标签
    # x_train, y_train = x_train[:, :dshape[1]], x_train[:, dshape[1]:]
    # x_test, y_test = x_test[:, :dshape[1]], x_test[:, dshape[1]:]
    # x_train_shuffle, y_train_shuffle = x_train_shuffle[:, :dshape[1]], x_train_shuffle[:, dshape[1]:]
    # x_test_shuffle, y_test_shuffle = x_test_shuffle[:, :dshape[1]], x_test_shuffle[:, dshape[1]:]

    # dataLoader for training
    # train_dataset = DatasetFolder(x_train)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers,
                                               pin_memory=pin_memory, drop_last=drop_last)
    # dataLoader for validating
    # test_dataset = DatasetFolder(x_test)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if shuffle:
        # train_shuffle_dataset = DatasetFolder(x_train_shuffle)
        train_shuffle_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_shuffle_loader = torch.utils.data.DataLoader(train_shuffle_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        # test_shuffle_dataset = DatasetFolder(x_test_shuffle)
        test_shuffle_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        test_shuffle_loader = torch.utils.data.DataLoader(test_shuffle_dataset, batch_size=batch_size,
                                                          shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


        return train_loader, test_loader, train_dataset, test_dataset,                train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

    return train_loader, test_loader, train_dataset, test_dataset

def load_data_from_txt(
        file_path,
        Ridx = 0, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
        evaluate = 1,
        shuffle = False,
        train_test_ratio=0.8,
        batch_size=30,
        num_workers=0,
        pin_memory=True,
        drop_last=True):

    print("loading data...")

    PathSet = {0:"./TestData", 1:"./CompetitionData1", 2:"./CompetitionData2", 3:"./CompetitionData3", 4:"./CompetitionData4"}
    PrefixSet = {0:"Test" , 1:"Round1", 2:"Round2", 3:"Round3", 4:"Round4"}

    PathRaw = "./chusai_data/" + PathSet[Ridx]
    PathOut = "./outputs/" + PathSet[Ridx]
    PathTrain = './train_data/' 
    Prefix = PrefixSet[Ridx]

    ## 1查找文件
    names= FindFiles(PathRaw) # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中

    dirs = os.listdir(PathRaw)
    names = []  # 文件编号
    files = []
    for f in sorted(dirs):
        if f.endswith('.txt'):
            files.append(f)
    for f in sorted(files):
        if f.find('CfgData')!=-1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])

    ## 2创建对象并处理
    Rst = []
    Gt  = []
    for na in names: #
    # for na in [names[0]]:#
        # 读取配置及CSI数据
        Cfg = CfgFormat(PathRaw + '/' + Prefix + 'CfgData' + na + '.txt')
        csi = np.genfromtxt(PathRaw + '/' + Prefix + 'InputData' + na + '.txt', dtype = float)
        CSI = csi[:,0::2] + 1j* csi[:,1::2]

        Nt = [0] + list(accumulate(Cfg['Nt']))

    data = np.load(file_path + '/CSI_train_conca.npy')
    dshape = data.shape
    data = data.astype('float32')
    label = np.load(file_path + '/GT_train.npy')
    label = label.astype('float32')
    # data_label = np.concatenate((data, label), axis=1)

    # if shuffle:
    #     data_copy = np.copy(data_label)
    #     np.random.shuffle(data_copy)
    #     data_shuffle = data_copy
        # data_transpose = data_copy.transpose()
        # np.random.shuffle(data_transpose)
        # data_shuffle = data_transpose.transpose()

    partition = int(data.shape[0] * train_test_ratio)
    x_train, x_test = data[:partition], data[partition:]
    y_train, y_test = label[:partition], label[partition:]
    # x_train_shuffle, x_test_shuffle = data_shuffle[:partition], data_shuffle[partition:]
    # 分离x_train, x_test的数据和标签
    # x_train, y_train = x_train[:, :dshape[1]], x_train[:, dshape[1]:]
    # x_test, y_test = x_test[:, :dshape[1]], x_test[:, dshape[1]:]
    # x_train_shuffle, y_train_shuffle = x_train_shuffle[:, :dshape[1]], x_train_shuffle[:, dshape[1]:]
    # x_test_shuffle, y_test_shuffle = x_test_shuffle[:, :dshape[1]], x_test_shuffle[:, dshape[1]:]

    # dataLoader for training
    # train_dataset = DatasetFolder(x_train)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers,
                                               pin_memory=pin_memory, drop_last=drop_last)
    # dataLoader for validating
    # test_dataset = DatasetFolder(x_test)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if shuffle:
        # train_shuffle_dataset = DatasetFolder(x_train_shuffle)
        train_shuffle_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_shuffle_loader = torch.utils.data.DataLoader(train_shuffle_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        # test_shuffle_dataset = DatasetFolder(x_test_shuffle)
        test_shuffle_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        test_shuffle_loader = torch.utils.data.DataLoader(test_shuffle_dataset, batch_size=batch_size,
                                                          shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


        return train_loader, test_loader, train_dataset, test_dataset,                train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

    return train_loader, test_loader, train_dataset, test_dataset
