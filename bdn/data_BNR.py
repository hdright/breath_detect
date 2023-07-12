# %%
import numpy as np
from scipy.signal import savgol_filter
from itertools import accumulate
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale, StandardScaler
import os
current_file_path = __file__
current_dir_path = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(current_dir_path, '../'))
from chusai import CsiFormatConvrt, FindFiles, CfgFormat
# from skimage import transform
# from estBreath import hampel, hampelpd
# from copy import deepcopy
# import random
import pandas as pd

# %%
PathSet = {0: "./TestData", 1: "./CompetitionData1", 2: "./CompetitionData2",
            3: "./CompetitionData3", 4: "./CompetitionData4"}
PrefixSet = {0: "Test", 1: "Round1", 2: "Round2", 3: "Round3", 4: "Round4"}

pi = np.pi

# 注意，csi的维度为M*N*T,rx*sc*T
def CSI_sanitization(csi):
    '''
    CSI线性相位去噪

    Parameters
    ----------
    csi : np.array
        CSI时域数据，维度为Nrx*Nsc*Nt

    Returns
    -------
    csi_phase : np.array
        CSI线性相位去噪后的数据，维度为Nrx*Nsc*Nt
    '''
    M, N, T = csi.shape  # 接收天线数量M
    fi = 1250 * 2  # 子载波间隔1250khz * 2
    csi_phase = np.zeros((M, N, T))
    for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有N个子载波
        csi_phase[0, :, t] = np.unwrap(np.angle(csi[0, :, t]))
        csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(csi[1, :, t] * np.conj(csi[0, :, t])))
        csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(csi[2, :, t] * np.conj(csi[1, :, t])))
        if M == 8:
            csi_phase[3, :, t] = np.unwrap(csi_phase[2, :, t] + np.angle(csi[3, :, t] * np.conj(csi[2, :, t])))
            csi_phase[4, :, t] = np.unwrap(csi_phase[3, :, t] + np.angle(csi[4, :, t] * np.conj(csi[3, :, t])))
            csi_phase[5, :, t] = np.unwrap(csi_phase[4, :, t] + np.angle(csi[5, :, t] * np.conj(csi[4, :, t])))
            csi_phase[6, :, t] = np.unwrap(csi_phase[5, :, t] + np.angle(csi[6, :, t] * np.conj(csi[5, :, t])))
            csi_phase[7, :, t] = np.unwrap(csi_phase[6, :, t] + np.angle(csi[7, :, t] * np.conj(csi[6, :, t])))
        ai = np.tile(2 * pi * fi * np.array(range(N)), M)
        bi = np.ones(M * N)
        if M == 3:
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]))
        elif M == 8:
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t], csi_phase[3, :, t],
                                 csi_phase[4, :, t], csi_phase[5, :, t], csi_phase[6, :, t], csi_phase[7, :, t]))
        else:
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t]))
            raise Exception("M must be 3 or 8")
        A = np.dot(ai, ai)
        B = np.dot(ai, bi)
        C = np.dot(bi, bi)
        D = np.dot(ai, ci)
        E = np.dot(bi, ci)
        rho_opt = (B * E - C * D) / (A * C - B ** 2)
        beta_opt = (B * D - A * E) / (A * C - B ** 2)
        temp = np.tile(np.array(range(N)), M).reshape(M, N)
        csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
    return csi_phase

def cal_BNR(dft, BPMrange, BPMresol):
    '''
    计算在BPMrange范围内的BNR，即dft的BPMrange范围内具有最大能量的 FFT bin，
    然后通过将该 bin 的能量除以所有 FFT bin 的能量总和来计算 BNR

    Parameters
    ----------
    dft : np.array
        m*n数组，为m行n点FFT投影的结果
    '''
    minIdx, maxIdx = int(BPMrange[0] / BPMresol), int(BPMrange[1] // BPMresol)
    # 能量等于dft的绝对值的平方的和
    dft_amp = np.abs(dft)
    dft_resp = dft_amp[:, minIdx:maxIdx]
    maxBin = np.argmax(dft_resp, axis=1)
    # 按行求BNR
    # BNR = (dft_resp[:, maxBin] ** 2) / np.sum(dft_amp ** 2, axis=1)
    BNR = np.array([dft_resp[i, maxBin[i]] ** 2 for i in range(dft.shape[0])]) / np.sum(dft_amp ** 2, axis=1)
    return BNR

def cal_gt_BNR(dft, gt, BPMresol, rangeGt=1):
    '''
    计算在BPMgroundTruth+-0.5范围内的BNR，即dft的BPMgroundTruth+-0.5范围内的能量之和，
    然后通过将该范围的能量除以所有 FFT bin 的能量总和来计算 BNR

    Parameters
    ----------
    dft : np.array
        m*n数组，为m行n点FFT投影的结果
    '''
    # rangeGt = 0.5
    dft_amp = np.abs(dft)
    if len(gt) == 1:
        minIdx, maxIdx = int((gt - rangeGt) / BPMresol), int((gt + rangeGt) / BPMresol)
        # 能量等于dft的绝对值的平方的和
        # dft_amp = np.abs(dft)
        dft_resp = dft_amp[:, minIdx:maxIdx]
        resp_energy = np.sum(dft_resp ** 2, axis=1)
    else:
        resp_energy = np.zeros(dft.shape[0])
        for i in range(len(gt)):
            minIdx, maxIdx = int((gt[i] - rangeGt) / BPMresol), int((gt[i] + rangeGt) / BPMresol)
            # 能量等于dft的绝对值的平方的和
            # dft_amp = np.abs(dft)
            dft_resp = dft_amp[:, minIdx:maxIdx]
            resp_energy += np.sum(dft_resp ** 2, axis=1)
        resp_energy /= len(gt)
    # 按行求BNR
    # BNR = (dft_resp[:, maxBin] ** 2) / np.sum(dft_amp ** 2, axis=1)
    BNR = resp_energy / np.sum(dft_amp ** 2, axis=1)
    return BNR

def matching_val_select(csi, nDft, dftSize, BPMrange=[6, 45], BPMresol=0.1):
    '''
    计算csi在角度0-99/50*pi上的投影，计算DFT，取使BNR最大的角度
    '''
    angMat = np.array([np.cos(np.linspace(0, 2 * np.pi, 100)),
                          np.sin(np.linspace(0, 2 * np.pi, 100))]) # shape:2*100
    # 将csi拆分为第一行为实部，第二行为虚部的矩阵
    csiMat = np.array([np.real(csi), np.imag(csi)]) # shape:2*Nt
    # 计算csi在角度0-99/50*pi上的投影
    csiProj = np.dot(angMat.T, csiMat)
    dftProj = np.fft.fft(csiProj, axis=1, n=nDft)[:, :dftSize]
    # 取使BNR最大的角度
    bnrMat = cal_BNR(dftProj, BPMrange=BPMrange, BPMresol=BPMresol)
    maxIdx = np.argmax(bnrMat)
    return dftProj[maxIdx, :]

def fft_shift_extend(data, shift, BPMrange, BPMresol):
    '''
    将x的minIndex, maxIndex区间的值循环右移，右移长度为shift/BPMresol个单位

    Parameters
    ----------
    data : np.ndarray
        输入数据，shape为(2, x, 600)
    shift : float
        右移长度，单位为BPM
    BPMrange : list
        BPM范围，单位为BPM
    BPMresol : float
        BPM分辨率，单位为BPM

    Returns
    -------
    x_delta : np.ndarray
        右移后的数据，shape为(2, x, 600)
    '''
    minBPM, maxBPM = BPMrange[0], BPMrange[1]
    minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
    # 将x的minIndex, maxIndex区间的值循环右移，右移长度为shift/BPMresol个单位
    shiftresol = int(shift / BPMresol)
    x_delta = data.copy()
    x_delta[:, :, minIndex:maxIndex] = np.roll(x_delta[:, :, minIndex:maxIndex], shiftresol, axis=2)
    return x_delta

def generate_gaussian_pulses(noBpmPoints, BPMresol, cfg, shift=0):
    '''
    生成高斯分布的ground truth

    Parameters
    ----------
    noBpmPoints : int
        BPM范围内的点数
    BPMresol : float
        BPM分辨率，单位为BPM
    cfg : dict
        从cfg文件中读取的配置信息
    shift : float, optional
        右移长度，单位为BPM, by default 0

    Returns
    -------
    gt_pd : np.ndarray
        高斯分布的ground truth，shape为(noBpmPoints,)
    '''
    gt_pd = np.zeros(noBpmPoints)
    sigma2 = (1 / BPMresol ** 2)
    x = np.arange(noBpmPoints)
    for p in range(cfg['Np']):
        # gt所有值的高斯分布叠加作为label
        mu = (cfg['gt'][p]+shift) / BPMresol
        gt_pd += np.exp(-(x - mu) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2) / cfg['Np']
    return gt_pd

allDataFormats = ['amp', 'diffPha', 'ampRatio', 'pha', 'sani', 'diffSani', 'ampRaBnr']

class BreathDataset(Dataset):
    def __init__(self, Round: int, 
                 path: str = "./chusai_data/", 
                 savePsdGtBnrPath: str = "./outputs/",
                 savePsdGtBnrName: str = "psd_gt_bnr.csv",
                 BPMresol: float = 1, 
                 breathEnd: int = 1, 
                 no_sample: int = 90,
                #  ampRatio: bool = False, # 是否采用CSI幅度比
                 dataBorrow: bool = False, # 3x30场景是否借用4x80场景数据进行训练
                 preProcList: list = ['amp', 'diffPha'], # 1\2种学习的数据分别用什么['amp', 'diffPha', 'ampRatio', 'pha']
                 pre_sg: list = [8, 7], # savgol_filter参数
                 bnr_range: list = [6, 45], # BNR计算范围
                 Np2extend: list = []): # 通过循环移位扩充3x30场景数据
        self.path = path
        self.round = Round
        if no_sample in [640, 180, 180640]:
            self.load_no_axis = 2
        elif no_sample in [960, 270, 270960]:
            self.load_no_axis = 3
        else:
            self.load_no_axis = 1
        if self.load_no_axis == 2:
            assert preProcList[0] in allDataFormats, 'preProcList[0] must be in ' + str(allDataFormats)
            assert preProcList[1] in allDataFormats, 'preProcList[1] must be in ' + str(allDataFormats)
            assert preProcList[0] != preProcList[1], 'preProcList[0] and preProcList[1] must be different'
        elif self.load_no_axis == 3:
            assert preProcList[0] in allDataFormats, 'preProcList[0] must be in ' + str(allDataFormats)
            assert preProcList[1] in allDataFormats, 'preProcList[1] must be in ' + str(allDataFormats)
            assert preProcList[2] in allDataFormats, 'preProcList[2] must be in ' + str(allDataFormats)
            assert preProcList[0] != preProcList[1], 'preProcList[0] and preProcList[1] must be different'
            assert preProcList[0] != preProcList[2], 'preProcList[0] and preProcList[2] must be different'
            assert preProcList[1] != preProcList[2], 'preProcList[1] and preProcList[2] must be different'
        else:
            assert preProcList[0] in allDataFormats, 'preProcList[0] must be in ' + str(allDataFormats)

        self.dataBorrow = dataBorrow and (Round == 0) and (no_sample % 90 == 0)
        self.preProcList = preProcList
        self.loadAmp = 'amp' in self.preProcList
        self.loadAmpRa = 'ampRatio' in self.preProcList
        self.loadDiffPha = 'diffPha' in self.preProcList
        self.loadPha = 'pha' in self.preProcList
        self.loadSani = 'sani' in self.preProcList
        self.loadDiffSani = 'diffSani' in self.preProcList
        self.loadAmpRaBnr = 'ampRaBnr' in self.preProcList
        self.savePsdGtBnrPath = savePsdGtBnrPath
        self.savePsdGtBnrName = savePsdGtBnrName

        PathRaw = os.path.join(path, PathSet[Round])
        Prefix = PrefixSet[Round]

        self.Np2extend = Np2extend
        # self.ampRatio = ampRatio
        savgol_window_length, savgol_polyorder = pre_sg[0], pre_sg[1]
        self.noSample = 0
        if Round == 0 and len(self.Np2extend): # 训练时通过循环移位扩展数据
            # self.extend_data = True
            print('Extending data of Np =', self.Np2extend)
        # else:
        #     self.extend_data = False
        self.BPMrange = [5, 50]  # 5-50BPM

        # self.names = FindFiles(PathRaw)  # 查找文件夹中包含的所有比赛/测试数据文件
        dirs = os.listdir(PathRaw)
        self.names = []  # 文件编号
        self.files = []
        for f in sorted(dirs):
            if f.endswith('.txt'):
                self.files.append(f)
        for f in sorted(self.files):
            if f.find('CfgData') != -1 and f.endswith('.txt'):
                self.names.append(f.split('CfgData')[-1].split('.txt')[0])

        # self.CSI_s = []
        self.CSI_fft = []
        self.gt_pd = []
        self.Cfg = []
        if no_sample % 90 == 0:
            if Round == 3:
                names = [self.names[0], self.names[3], self.names[4]]
            elif Round == 2:
                names = [self.names[0], self.names[3]]
            else:
                # names = [self.names[0], self.names[9]]
                if self.dataBorrow:
                    names = self.names[:9]
                else:
                    names = [self.names[0]]   
        elif no_sample % 320 == 0:
            if Round == 3 or Round == 2:
                names = self.names[1:3]
            else:
                names = self.names[1:9]
        elif no_sample in [90320,180640]: # 将90*600的fft矩阵扩展为320*600
            if Round == 3 or Round == 2:
                names = self.names
            else:
                names = self.names[0:9]
        else:
            names = []
            raise ValueError('no_sample should be 90 or 320')
        ampScalar = StandardScaler()
        # phaScalar = StandardScaler()
        # axis0FftScaler = StandardScaler()
        # axis1FftScaler = StandardScaler()
        # axis2FftScaler = StandardScaler()
        psd_gt_BNR = np.zeros((6, 6)) # 第二个维度rangeGt分别为0.5,1，2，3，4，5
        rangeGtList = [0.5, 1, 2, 3, 4, 5]
        for na in names:  
            print("Loading %s InputData %s / %d" % (Prefix, na, len(self.names)))
            # print("%s / %d" % (na, len(self.names)))
            # 读取配置及CSI数据
            Cfg = CfgFormat(PathRaw + '/' + Prefix + 'CfgData' + na + '.txt')
            csi = np.genfromtxt(PathRaw + '/' + Prefix + 'InputData' + na + '.txt', dtype=float)
            CSI = csi[:, 0::2] + 1j * csi[:, 1::2]
            
            Nt = [0] + list(accumulate(Cfg['Nt']))
            for ii in range(Cfg['Nsamp']):
                cfg = {
                    'Np': Cfg['Np'][ii],
                    'Ntx': Cfg['Ntx'],
                    'Nrx': Cfg['Nrx'],
                    'Nsc': Cfg['Nsc'],
                    'Nt': Cfg['Nt'][ii],
                    'Tdur': Cfg['Tdur'][ii],
                    'fstart': Cfg['fstart'],
                    'fend': Cfg['fend'],
                    'na': na
                }
                # 3x30场景加载8x40的CSI数据
                if self.dataBorrow and int(na) > 1:
                    cfg['Nrx'] = 3
                    cfg['Nsc'] = 30
                if self.round == 0:
                    with open(PathRaw + '/' + Prefix + 'GroundTruthData' + na + '.txt', 'r') as f:
                        gt = [np.fromstring(arr.strip(), dtype=float, sep = ' ') for arr in f.readlines()]
                    cfg['gt'] = np.pad(gt[ii], (0, 3 - len(gt[ii])), 'constant') # 补齐3个值, 防止dataloader因长度不一致报错
                self.Cfg.append(cfg)
                self.noSample += 1 # 每添加一个cfg，noSample加1
                # self.CSI_s.append(CsiFormatConvrt(CSI[Nt[ii]:Nt[ii+1], :], Cfg['Nrx'],
                #                                   Cfg['Ntx'], Cfg['Nsc'], Cfg['Nt'][ii]))
                CSI_sam = CsiFormatConvrt(CSI[Nt[ii]:Nt[ii+1], :], Cfg['Nrx'],
                                                  Cfg['Ntx'], Cfg['Nsc'], Cfg['Nt'][ii])
                # 3x30场景加载8x40的CSI数据
                if self.dataBorrow and int(na) > 1:
                    # 只保留CSI_sam第0维度的3:6个值，第1维度的所有值，第2维度的5:35个值，第3维度的所有值
                    CSI_sam = CSI_sam[3:6, :, 5:35, :]
                fs = cfg['Nt'] / cfg['Tdur']  # 采样频率
                # BPMresol = 0.1
                resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
                Ndft = int(fs / resol)  # DFT点数

                # breathEnd = 1  # 呼吸最高频率 Hz
                dftSize = int(breathEnd / resol)  # DFT宽度

                # 要估计的呼吸频率范围
                bpmRange = np.arange(0, 60, BPMresol)
                noBpmPoints = len(bpmRange)  # 要估计的呼吸频率个数


                if self.loadAmp:
                    csiAmpFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiAmpFft = np.zeros(1)
                if self.loadAmpRa:
                    csiAmpRaFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiAmpRaFft = np.zeros(1)
                if self.loadPha:
                    csiPhaFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiPhaFft = np.zeros(1)
                if self.loadDiffPha:
                    csiDiffPhaFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiDiffPhaFft = np.zeros(1)
                if self.loadSani or self.loadDiffSani:
                    csi_sanitization = CSI_sanitization(np.squeeze(CSI_sam))
                else:
                    csi_sanitization = np.zeros(1)
                if self.loadSani:
                    csiSaniFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiSaniFft = np.zeros(1)
                if self.loadDiffSani:
                    csiDiffSaniFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiDiffSaniFft = np.zeros(1)
                if self.loadAmpRaBnr:
                    csiAmpRaBnrFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiAmpRaBnrFft = np.zeros(1)
                j = 0
                for i in range(cfg['Nrx']):
                    for k in range(cfg['Nsc']):
                        # ampFiltered = savgol_filter(hampelpd(np.abs(CSI_sam[i, j, k, :]), 400), savgol_window_length, savgol_polyorder) # hampel+savgol性能很差
                        if self.loadAmpRa:
                            ampFiltered = scale(savgol_filter(np.abs(CSI_sam[i, j, k, :]/
                                                            np.where(CSI_sam[(i+1)%cfg['Nrx'], j, k, :] == 0, 1, CSI_sam[(i+1)%cfg['Nrx'], j, k, :])), 
                                                            savgol_window_length, savgol_polyorder))
                            csiAmpRaFft[i * cfg['Nsc'] + k] = np.fft.fft(ampFiltered, Ndft)[0:dftSize] #
                        if self.loadAmp:
                            ampFiltered = savgol_filter(np.abs(CSI_sam[i, j, k, :]), savgol_window_length, savgol_polyorder)
                            if no_sample % 90 != 0: # 4x80场景统一scale时域幅度 TODO
                                if (j, k) == (0, 0):
                                    ampScalar.fit(ampFiltered.reshape(-1, 1))
                                csiAmpScaled = ampScalar.transform(ampFiltered.reshape(-1, 1)).reshape(-1)
                                # csiAmpScaled = ampScalar.fit_transform(ampFiltered.reshape(-1, 1)).reshape(-1) # 不统一不对
                                csiAmpFft[i * cfg['Nsc'] + k] = np.fft.fft(csiAmpScaled, Ndft)[0:dftSize]
                            else: # 3x30场景不scale时域幅度
                                # csiAmpFft[i * cfg['Nsc'] + k] = np.fft.fft(scale(ampFiltered), Ndft)[0:dftSize]
                                csiAmpFft[i * cfg['Nsc'] + k] = np.fft.fft(ampFiltered, Ndft)[0:dftSize]
                        if self.loadPha:
                            phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]), savgol_window_length, savgol_polyorder)
                            # 独立std
                            phaScaled = scale(phaFiltered)
                            csiPhaFft[i * cfg['Nsc'] + k] = np.fft.fft(phaScaled, Ndft)[0:dftSize]
                        if self.loadDiffPha:
                            phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]) -
                                                            np.angle(CSI_sam[(i+1)%cfg['Nrx'], j, k, :]), 
                                                            savgol_window_length, savgol_polyorder)
                            # if i>0:
                            #     phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]) -
                            #                                 np.angle(CSI_sam[(i+1)%cfg['Nrx'], j, k, :]), savgol_window_length, savgol_polyorder)
                            # else:
                            #     phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]), savgol_window_length, savgol_polyorder)
                            # 独立std
                            phaScaled = scale(phaFiltered)
                            csiDiffPhaFft[i * cfg['Nsc'] + k] = np.fft.fft(phaScaled, Ndft)[0:dftSize]
                        if self.loadSani:
                            csiSaniFft[i * cfg['Nsc'] + k] = np.fft.fft(scale(savgol_filter(csi_sanitization[i][k], 
                                                                                            savgol_window_length, savgol_polyorder)), Ndft)[:dftSize]
                        if self.loadDiffSani:
                            csiDiffSaniFft[i * cfg['Nsc'] + k] = np.fft.fft(scale(savgol_filter(
                                csi_sanitization[i][k] - csi_sanitization[(i+1)%cfg['Nrx']][k], 
                                savgol_window_length, savgol_polyorder)), Ndft)[:dftSize]
                        if self.loadAmpRaBnr:
                            csi_ratio_amp = scale(savgol_filter(np.abs(CSI_sam[i, j, k, :]/
                                                                        np.where(CSI_sam[(i+1)%cfg['Nrx'], j, k, :] == 0, 
                                                                                1, CSI_sam[(i+1)%cfg['Nrx'], j, k, :])), 
                                                                savgol_window_length, savgol_polyorder))
                            csi_ratio_phase = np.angle(CSI_sam[i, j, k, :]) - np.angle(CSI_sam[(i+1)%cfg['Nrx'], j, k, :])
                            csi_ratio = csi_ratio_amp * np.exp(1j * csi_ratio_phase)
                            csiAmpRaBnrFft[i * cfg['Nsc'] + k] = matching_val_select(csi_ratio, Ndft, dftSize, BPMrange=bnr_range)
                for range_idx, rangeGT in enumerate(rangeGtList):
                    if self.loadAmp:
                        psd_gt_BNR[0, range_idx] += np.sum(cal_gt_BNR(csiAmpFft, gt[ii], BPMresol, rangeGT))
                    if self.loadAmpRa:
                        psd_gt_BNR[1, range_idx] += np.sum(cal_gt_BNR(csiAmpRaFft, gt[ii], BPMresol, rangeGT))
                    if self.loadPha:
                        psd_gt_BNR[2, range_idx] += np.sum(cal_gt_BNR(csiPhaFft, gt[ii], BPMresol, rangeGT))
                    if self.loadDiffPha:
                        psd_gt_BNR[3, range_idx] += np.sum(cal_gt_BNR(csiDiffPhaFft, gt[ii], BPMresol, rangeGT))
                    if self.loadSani:
                        psd_gt_BNR[4, range_idx] += np.sum(cal_gt_BNR(csiSaniFft, gt[ii], BPMresol, rangeGT))
                    if self.loadDiffSani:
                        psd_gt_BNR[5, range_idx] += np.sum(cal_gt_BNR(csiDiffSaniFft, gt[ii], BPMresol, rangeGT))
                # if self.loadAmpRaBnr:
                #     psd_gt_BNR[6] += np.sum(cal_gt_BNR(csiAmpRaBnrFft, gt[ii], BPMresol))
                # if self.loadAmpRa:
                #     csiAmpRaFft = np.abs(csiAmpRaFft)
                # if self.loadAmp:
                #     csiAmpFft = np.abs(csiAmpFft)
                # if self.loadPha:
                #     csiPhaFft = np.abs(csiPhaFft)
                # if self.loadDiffPha:
                #     csiDiffPhaFft = np.abs(csiDiffPhaFft)
                # if self.loadSani:
                #     csiSaniFft = np.abs(csiSaniFft)
                # if self.loadDiffSani:
                #     csiDiffSaniFft = np.abs(csiDiffSaniFft)
                # if self.loadAmpRaBnr:
                #     csiAmpRaBnrFft = np.abs(csiAmpRaBnrFft)
                

            del CSI
        # 保存psd_gt_BNR至csv文件,列标注为rangeGtList，行标注为['Amp', 'AmpRa', 'Pha', 'DiffPha', 'Sani', 'DiffSani'] 
        psd_gt_BNR = pd.DataFrame(psd_gt_BNR, columns=rangeGtList, index=['Amp', 'AmpRa', 'Pha', 'DiffPha', 'Sani', 'DiffSani'])
        psd_gt_BNR.to_csv(os.path.join(self.savePsdGtBnrPath, self.savePsdGtBnrName),
                            index=False, encoding='utf-8')
        print('数据加载完成！总共加载了{}个样本'.format(self.noSample))
            

    def __len__(self) -> int:
        return len(self.CSI_fft)

    def __getitem__(self, idx) -> tuple[list, list, dict]|tuple[list, dict]:
        if self.round == 0:
            return self.CSI_fft[idx], self.gt_pd[idx], self.Cfg[idx]
        else:
            return self.CSI_fft[idx], self.Cfg[idx]
        

if __name__ == '__main__':
    BreathDataset(0, BPMresol=0.1, no_sample=270, Np2extend=[], 
                    pre_sg=[5, 3], preProcList=['amp', 'diffPha', 'ampRatio', 'pha', 'sani', 'diffSani'], dataBorrow=False, 
                    bnr_range = [6, 45], savePsdGtBnrName="psd_gt_bnr_001.csv")
    BreathDataset(0, BPMresol=0.1, no_sample=960, Np2extend=[],
                    pre_sg=[5, 3], preProcList=['amp', 'diffPha', 'ampRatio', 'pha', 'sani', 'diffSani'], dataBorrow=False,
                    bnr_range=[6, 45], savePsdGtBnrName="psd_gt_bnr_002to009.csv")