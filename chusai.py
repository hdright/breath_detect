'''
@file chusai.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-06-08 12:08:24
@modified: 2023-06-08 19:04:47
'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
修改输出格式为RoundYOutputX.txt, 2023年5月4日09:55:03
"""
import os, time
import numpy as np
from itertools import accumulate
from estBreath import estChusai
#numpy 1.19


def EstBreathRate(Cfg, CSI, iSamp = 0):
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    #########以下代码，参赛者用自己代码替代################
    #########样例代码中直接返回随机数作为估计结果##########
    result = np.random.rand(Cfg['Np'][iSamp]) * 45 + 5
    result = np.sort(result)
    return result

def RMSEerr(EstIn, GtIn):
    '''
    计算RMSE误差
    :param Est: 估计的呼吸率，1D
    :param Gt: 测量的呼吸率，1D
    :return: rmse误差
    '''
    Est = np.concatenate(EstIn)
    Gt = np.concatenate(GtIn)
    if np.size(Est) != np.size(Gt):
        print("呼吸率估计数目有误，输出无效!")
        return -1
    rmse = np.sqrt(np.mean(np.square(Gt - Est)))
    return rmse

def CsiFormatConvrt(Hin, Nrx, Ntx, Nsc, Nt):
    '''
    csi格式转换，从2D [NT x (Nsc*NRx*NTx)]转为4D [NRx][NTx][NSc][NT]
    '''
    Hout = np.reshape(Hin, [Nt, Nsc, Nrx, Ntx])
    Hout = np.transpose(Hout, [2, 3, 1, 0])
    return Hout

def EstRRByWave(wa, fs):
    Rof = 3
    n = 2**(np.ceil(np.log2(len(wa)))+Rof)
    blow, bhigh = [5, 50] # 约定呼吸率区间
    low = int(np.ceil(blow/60/fs*n))
    high = int(np.floor(bhigh/60/fs*n))
    spec = abs(np.fft.fft(wa-np.mean(wa), int(n)))
    tap = np.argmax(spec[low: high]) + low

    return tap/n*fs*60

class SampleSet:
    "样本集基类"
    Nsamples = 0 #总样本数类变量

    def __init__(self, name, Cfg, CSIs):
        self.name  = name
        self.Cfg   = Cfg
        self.CSI   = CSIs #所有CSI
        self.CSI_s = []   #sample级CSI
        self.Rst   = []
        self.Wave  = []   # 测量所得呼吸波形，仅用于测试
        self.Gt    = []   # 测量呼吸率，仅用于测试
        self.GtRR  = []   # 测量波形呼吸率，仅用于测试
        SampleSet.Nsamples += self.Cfg['Nsamp']

    def estBreathRate(self):
        BR = []
        # CSI数据整形，建议参赛者根据算法方案和编程习惯自行设计，这里按照比赛说明书将CSI整理成4D数组，4个维度含义依次为收天线，发天线，子载波，时间域测量索引
        Nt = [0] + list(accumulate(self.Cfg['Nt']))
        for ii in range(self.Cfg['Nsamp']):
            self.CSI_s.append(CsiFormatConvrt(self.CSI[Nt[ii]:Nt[ii+1],:], self.Cfg['Nrx'],
                                              self.Cfg['Ntx'], self.Cfg['Nsc'], self.Cfg['Nt'][ii]))
        for ii in range(self.Cfg['Nsamp']):
            # br = EstBreathRate(self.Cfg, self.CSI_s[ii], ii)  ## 呼吸率估计
            # 如不存在CSI_s[ii]的npy文件，则保存
            if not os.path.exists('/data/hdr/breath_detect/CSI_s_{}.npy'.format(ii)) and ii<10:
                np.save('/data/hdr/breath_detect/CSI_s_{}.npy'.format(ii), self.CSI_s[ii])
            br = estChusai(self.Cfg, self.CSI_s[ii], ii)  ## 呼吸率估计
            BR.append(br)
        self.Rst = BR

    def getRst(self):
        return self.Rst

    def getEstErr(self):
        rmseE = RMSEerr(self.Rst, self.Gt)
        print("<<<RMSE Error of SampleSet file #{} is {}>>>\n".format(self.name, rmseE))
        return rmseE

    def setGt(self, Gt):
        self.Gt = Gt

    def setWave(self, wave):
        #此处按照样例排布Wave波形，如self.Wave[iSamp][iPerson]['Wave'] 第iSamp个样例的第iPerson个人的波形
        NP = [0] + list(accumulate(self.Cfg['Np']))
        for ii in range(self.Cfg['Nsamp']):
            self.Wave.append(wave[NP[ii]:NP[ii+1]])

    def estRRByWave(self):
        for ii in range(len(self.Wave)):
            RR = []
            for jj in range(len(self.Wave[ii])):
                wa = abs(self.Wave[ii][jj]['Wave'])
                para = self.Wave[ii][jj]['Param']
                fs = (para[0]-1) / para[1]
                RR.append(EstRRByWave(wa, fs))
            #print("rr = ", RR, ", ii ", ii, ", jj", jj)
            self.GtRR.append(np.sort(np.array(RR)))
        return self.GtRR




def FindFiles(PathRaw):
    dirs = os.listdir(PathRaw)
    names = []  #文件编号
    files = []
    for f in sorted(dirs):
        if f.endswith('.txt'):
            files.append(f)
    for f in sorted(files):
        if f.find('CfgData')!= -1 and f.endswith('.txt'):
            print('Now reading file {} ...\n'.format(f))
            names.append(f.split('CfgData')[-1].split('.txt')[0])
    return names, files

def CfgFormat(fn):
    a = []
    with open(fn, 'r') as f:
        for line in f:
            d = np.fromstring(line, dtype = float, sep = ' ')#[0]
            a.append(d)
    return {'Nsamp':int(a[0][0]), 'Np': np.array(a[1],'int'), 'Ntx':int(a[2][0]), 'Nrx':int(a[3][0]),
            'Nsc':int(a[4][0]), 'Nt':np.array(a[5],'int'), 'Tdur':a[6], 'fstart':a[7][0], 'fend':a[8][0]}

def ReadWave(fn):
    Wave = []
    with open(fn, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            wa = {}
            wa['Param'] = np.fromstring(lines[i].strip(), dtype=float, sep = ' ')
            wa['Wave'] = np.fromstring(lines[i+1].strip(), dtype=int, sep = ' ')
            Wave.append(wa)
    return Wave

if __name__ == "__main__":
    print("<<< 水之波动 一之型 >>>\n")
    ## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义
    PathSet = {0:"./TestData", 1:"./CompetitionData1", 2:"./CompetitionData2", 3:"./CompetitionData3", 4:"./CompetitionData4"}
    PrefixSet = {0:"Test" , 1:"Round1", 2:"Round2", 3:"Round3", 4:"Round4"}

    Ridx = 0 # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = "./chusai_data/" + PathSet[Ridx]
    PathOut = "./outputs/" + PathSet[Ridx]
    Prefix = PrefixSet[Ridx]

    tStart = time.perf_counter()
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
    for na in names: #[names[0]]:#
        # 读取配置及CSI数据
        Cfg = CfgFormat(PathRaw + '/' + Prefix + 'CfgData' + na + '.txt')
        csi = np.genfromtxt(PathRaw + '/' + Prefix + 'InputData' + na + '.txt', dtype = float)
        CSI = csi[:,0::2] + 1j* csi[:,1::2]

        samp = SampleSet(na, Cfg, CSI)
        del CSI

        # 计算并输出呼吸率
        samp.estBreathRate()  ## 请进入该函数以找到编写呼吸率估计算法的位置
        rst = samp.getRst()
        Rst.extend(rst)

        # 3输出结果：各位参赛者注意输出值的精度
        if os.path.exists(PathOut) == False:
            os.makedirs(PathOut)
        with open(PathOut + '/' + Prefix + 'OutputData' + na + '.txt', 'w') as f:
            [np.savetxt(f, np.array(ele).reshape(1, -1), fmt = '%.6f', newline = '\n') for ele in rst]

        # 对于测试数据，参赛选手可基于真实呼吸数据计算估计RMSE
        if Ridx == 0:
            with open(PathRaw + '/' + Prefix + 'GroundTruthData' + na + '.txt', 'r') as f:
                 gt = [np.fromstring(arr.strip(), dtype=float, sep = ' ') for arr in f.readlines()]
            samp.setGt(gt)
            samp.getEstErr()  ## 计算每个输入文件的RMSE
            Gt.extend(gt)
        if Ridx == 0: ## 对于测试数据，参赛选手可以读取真实呼吸波形用于分析
            Wave = ReadWave(PathRaw + '/' + Prefix + 'BreathWave' + na + '.txt')
            samp.setWave(Wave)
            samp.estRRByWave() # 从依赖波形中计算呼吸率

    if Ridx == 0: # 对于测试数据，计算所有样本的RMSE
        rmseAll = RMSEerr(Rst, Gt)
        print("<<<RMSE Error of all Samples is {}>>>\n".format(rmseAll))


    ## 4统计时间
    tEnd = time.perf_counter()
    print("Total time consuming = {}s".format(round(tEnd-tStart, 3)))

