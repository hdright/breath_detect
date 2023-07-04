# %%
import numpy as np
from scipy.signal import savgol_filter
from itertools import accumulate
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale, StandardScaler
from chusai import CsiFormatConvrt, FindFiles, CfgFormat
import os
from skimage import transform
from estBreath import hampel, hampelpd
from copy import deepcopy
import random

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

allDataFormats = ['amp', 'diffPha', 'ampRatio', 'pha', 'diffSani', 'ampRaBnr']

class BreathDataset(Dataset):
    def __init__(self, Round: int, 
                 path: str = "./chusai_data/", 
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
        self.loadDiffSani = 'diffSani' in self.preProcList
        self.loadAmpRaBnr = 'ampRaBnr' in self.preProcList

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
        axis0FftScaler = StandardScaler()
        axis1FftScaler = StandardScaler()
        axis2FftScaler = StandardScaler()
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
                # if self.load_2axis:
                if self.loadDiffPha:
                    csiDiffPhaFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csiDiffPhaFft = np.zeros(1)
                if self.loadDiffSani:
                    csi_sanitization = CSI_sanitization(np.squeeze(CSI_sam))
                    csiDiffSaniFft = np.zeros((cfg['Nrx'] * cfg['Ntx'] * cfg['Nsc'], dftSize), dtype=complex)
                else:
                    csi_sanitization = np.zeros(1)
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
                if self.loadAmpRa:
                    csiAmpRaFft = np.abs(csiAmpRaFft)
                if self.loadAmp:
                    csiAmpFft = np.abs(csiAmpFft)
                if self.loadDiffPha:
                    csiDiffPhaFft = np.abs(csiDiffPhaFft)
                if self.loadDiffSani:
                    csiDiffSaniFft = np.abs(csiDiffSaniFft)
                # if no_sample == 90320 or no_sample == 90:
                match self.preProcList[0]:
                    case 'amp':
                        csiAxis0Fft = csiAmpFft
                    case 'ampRatio':
                        csiAxis0Fft = csiAmpRaFft
                    case 'diffPha':
                        csiAxis0Fft = csiDiffPhaFft
                    # case 'pha':
                    #     pass
                    case 'diffSani':
                        csiAxis0Fft = csiDiffSaniFft
                    case 'ampRaBnr':
                        csiAxis0Fft = csiAmpRaBnrFft
                    case _:
                        csiAxis0Fft = csiDiffPhaFft
                        raise ValueError('preProcList[0] error')
                if len(self.preProcList) > 1:
                    match self.preProcList[1]:
                        case 'amp':
                            csiAxis1Fft = csiAmpFft
                        case 'ampRatio':
                            csiAxis1Fft = csiAmpRaFft
                        case 'diffPha':
                            csiAxis1Fft = csiDiffPhaFft
                        # case 'pha':
                        #     pass
                        case 'diffSani':
                            csiAxis1Fft = csiDiffSaniFft
                        case 'ampRaBnr':
                            csiAxis1Fft = csiAmpRaBnrFft
                        case _:
                            csiAxis1Fft = csiDiffPhaFft
                            raise ValueError('preProcList[1] error')
                else:
                    csiAxis1Fft = csiAxis0Fft
                if len(self.preProcList) > 2:
                    match self.preProcList[2]:
                        case 'amp':
                            csiAxis2Fft = csiAmpFft
                        case 'ampRatio':
                            csiAxis2Fft = csiAmpRaFft
                        case 'diffPha':
                            csiAxis2Fft = csiDiffPhaFft
                        # case 'pha':
                        #     pass
                        case 'diffSani':
                            csiAxis2Fft = csiDiffSaniFft
                        case 'ampRaBnr':
                            csiAxis2Fft = csiAmpRaBnrFft
                        case _:
                            csiAxis2Fft = csiDiffPhaFft
                            raise ValueError('preProcList[2] error')
                else:
                    csiAxis2Fft = csiAxis0Fft
                if no_sample == 180640:
                    if Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'] == 90:
                        # 将90*600的csiAmpFft和csiPhaFft矩阵扩展为320*600
                        CsiAxis0StretchSc = np.zeros((120, noBpmPoints))
                        CsiAxis0StretchRx = np.zeros((320, noBpmPoints))
                        CsiAxis1StretchSc = np.zeros((120, noBpmPoints))
                        CsiAxis1StretchRx = np.zeros((320, noBpmPoints))
                        # 每30行拉伸为40行
                        for i in range(3):
                            CsiAxis0StretchSc[i*40:(i+1)*40, :] = transform.resize(csiAxis0Fft[i*30:(i+1)*30, :], (40, noBpmPoints), order=3)
                            CsiAxis1StretchSc[i*40:(i+1)*40, :] = transform.resize(csiAxis1Fft[i*30:(i+1)*30, :], (40, noBpmPoints), order=3)
                        # 120行中，每隔40行取一行，一共能取3行（如0、40、80，等差数列），拉伸为8行，形成320行
                        for i in range(40):
                            sc_idx = np.arange(i, 120, 40)
                            CsiAxis0StretchRx[i*8:(i+1)*8, :] = transform.resize(CsiAxis0StretchSc[sc_idx, :], (8, noBpmPoints), order=3)
                            CsiAxis1StretchRx[i*8:(i+1)*8, :] = transform.resize(CsiAxis1StretchSc[sc_idx, :], (8, noBpmPoints), order=3)
                        csiAxis0Fft = CsiAxis0StretchRx
                        csiAxis1Fft = CsiAxis1StretchRx
                elif no_sample == 90320:
                    if Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'] == 90:
                        # 将90*600的fft矩阵扩展为320*600
                        CSIfft_stretch_sc = np.zeros((120, noBpmPoints))
                        CSIfft_stretch_rx = np.zeros((320, noBpmPoints))
                        # 每30行拉伸为40行
                        for i in range(3):
                            CSIfft_stretch_sc[i*40:(i+1)*40, :] = transform.resize(csiAxis0Fft[i*30:(i+1)*30, :], (40, noBpmPoints), order=3)
                        # 120行中，每隔40行取一行，一共能取3行（如0、40、80，等差数列），拉伸为8行，形成320行
                        for i in range(40):
                            sc_idx = np.arange(i, 120, 40)
                            rx_idx = np.arange(i, 320, 40)
                            CSIfft_stretch_rx[rx_idx, :] = transform.resize(CSIfft_stretch_sc[sc_idx, :], (8, noBpmPoints), order=3)
                        csiAxis0Fft = CSIfft_stretch_rx
                        
                # CSIfft按行标准化
                csiAxis0Fft = axis0FftScaler.fit_transform(csiAxis0Fft.T).T
                # 在现有的0axis之前创建一个新axis，并组合csiAmpFft和csiPhaFft
                csiAxis0Fft = np.expand_dims(csiAxis0Fft, axis=0)
                if len(self.preProcList) > 1:
                    csiAxis1Fft = axis1FftScaler.fit_transform(csiAxis1Fft.T).T
                    csiAxis1Fft = np.expand_dims(csiAxis1Fft, axis=0)
                    csiFft = np.concatenate((csiAxis0Fft, csiAxis1Fft), axis=0)
                else:
                    csiFft = csiAxis0Fft
                if len(self.preProcList) > 2:
                    csiAxis2Fft = axis2FftScaler.fit_transform(csiAxis2Fft.T).T
                    csiAxis2Fft = np.expand_dims(csiAxis2Fft, axis=0)
                    csiFft = np.concatenate((csiFft, csiAxis2Fft), axis=0)
                else:
                    csiFft = csiAxis0Fft
                # # CSIfft按列标准化
                # csiAxis0Fft = axis0FftScaler.fit_transform(csiAxis0Fft)
                # csiAxis1Fft = axis1FftScaler.fit_transform(csiAxis1Fft)
                
                if self.load_no_axis > 1:
                    self.CSI_fft.append(csiFft.astype(np.float32))
                else:
                    self.CSI_fft.append(csiAxis0Fft.astype(np.float32))

                # 生成标签数据，是cfg['gt']的概率的类one-hot编码
                if self.round == 0:
                    gt_pd = generate_gaussian_pulses(noBpmPoints, BPMresol, cfg)
                    self.gt_pd.append(gt_pd)

                    if cfg['Np'] in self.Np2extend:
                        if cfg['Np'] in [2, 3]:
                            # 2,3之间的随机小数保留一位有效数字
                            shift_interval = 2
                            # 左移次数，左移最低到8bpm
                            leftShiftTimes = int(np.floor((min(cfg['gt']) - 8) / shift_interval))
                            # 右移次数，右移最多floor(10/shift_interval)次，最高到45bpm
                            rightShiftTimes = int(min(np.floor(10 / shift_interval), 
                                                np.floor((45 - max(cfg['gt'])) / shift_interval)))
                        else:
                            # 4,6之间的随机小数保留一位有效数字
                            shift_interval = round(random.uniform(4, 6), 1)
                            # 左移次数，左移最低到8bpm，最多一次
                            leftShiftTimes = int(min(np.floor((min(cfg['gt']) - 8) / shift_interval), 1))
                            # 右移次数，右移最多floor(10/shift_interval)次，最高到45bpm
                            rightShiftTimes = int(min(np.floor(10 / shift_interval), 
                                                np.floor((45 - max(cfg['gt'])) / shift_interval)))
                        # 左移fft
                        for i in range(leftShiftTimes):
                            shift = shift_interval * (i + 1)
                            # 生成扩展的标签数据
                            cfgcopy = deepcopy(cfg)
                            cfgcopy['gt'] = cfgcopy['gt'] - shift
                            self.noSample += 1 # 每添加一个cfg，noSample加1
                            self.Cfg.append(cfgcopy)
                            gt_pd = generate_gaussian_pulses(noBpmPoints, BPMresol, cfgcopy)
                            self.gt_pd.append(gt_pd)
                            # 生成扩展的csi数据
                            csiFftEx = fft_shift_extend(csiFft.astype(np.float32), -shift, 
                                                        self.BPMrange, BPMresol)
                            self.CSI_fft.append(csiFftEx)
                        # 右移fft
                        for i in range(rightShiftTimes):
                            shift = shift_interval * (i + 1)
                            # 生成扩展的标签数据
                            cfgcopy = deepcopy(cfg)
                            cfgcopy['gt'] = cfgcopy['gt'] + shift
                            self.noSample += 1 # 每添加一个cfg，noSample加1
                            self.Cfg.append(cfgcopy)
                            gt_pd = generate_gaussian_pulses(noBpmPoints, BPMresol, cfgcopy)
                            self.gt_pd.append(gt_pd)
                            # 生成扩展的csi数据
                            csiFftEx = fft_shift_extend(csiFft.astype(np.float32), shift, 
                                                        self.BPMrange, BPMresol)
                            self.CSI_fft.append(csiFftEx)

            del CSI
        print('数据加载完成！总共加载了{}个样本'.format(self.noSample))
            

    def __len__(self) -> int:
        return len(self.CSI_fft)

    def __getitem__(self, idx) -> tuple[list, list, dict]|tuple[list, dict]:
        if self.round == 0:
            return self.CSI_fft[idx], self.gt_pd[idx], self.Cfg[idx]
        else:
            return self.CSI_fft[idx], self.Cfg[idx]
        


def load_data_from_txt(
        Ridx = 0, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
        no_sample = 90, # 设置样本数
        dataBorrow: bool = False, # 3x30场景是否借用4x80场景数据进行训练
        preProcList = ['amp', 'diffPha'], # 1\2种学习的数据分别用什么
        pre_sg = [8, 7], # 设置预处理滤波器参数
        Np2extend = [], # 设置需要扩展的Np
        BPMresolution = 1.0, # 设置BPM分辨率
        batch_size = 1, # 设置batch大小
        shuffle = True, # 设置是否打乱数据
        num_workers = 0, # 设置读取数据的线程数量
        bnr_range = [6, 45] # 设置BNR读取的bin范围
):
    """
    从txt文件中读取数据
    
    Parameters
    ----------
    Ridx : int, optional
        比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ..., by default 0
    BPMresolution : float, optional
        设置BPM分辨率, by default 1
    batch_size : int, optional
        设置batch大小, by default 1
    shuffle : bool, optional
        设置是否打乱数据, by default True
    num_workers : int, optional
        设置读取数据的线程数量, by default 0
    """
    train_set = BreathDataset(Ridx, BPMresol=BPMresolution, no_sample=no_sample, Np2extend=Np2extend, 
                              pre_sg=pre_sg, preProcList=preProcList, dataBorrow=dataBorrow, 
                              bnr_range = bnr_range)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader



def save_data_to_txt(
        rst: list, # 设置数据
        file_na: list, # 设置文件名
        Ridx = 0, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
        path: str = "./outputs/", # 设置数据存放目录
        # BPMresolution = 1.0, # 设置BPM分辨率
        # batch_size = 1, # 设置batch大小
        # shuffle = True, # 设置是否打乱数据
        # num_workers = 0, # 设置读取数据的线程数量
):
    """
    向txt文件中保存数据
    
    Parameters
    ----------
    rst : list
        设置数据
    file_na : str
        设置文件名
    Ridx : int, optional
        比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ..., by default 0
    path : str, optional
        设置数据存放目录, by default "./outputs/"
    """

    PathOut = path + PathSet[Ridx]
    Prefix = PrefixSet[Ridx]

    # 输出结果：
    if os.path.exists(PathOut) == False:
        os.makedirs(PathOut)
    with open(PathOut + '/' + Prefix + 'OutputData' + file_na[0] + '.txt', 'w') as f:
        [np.savetxt(f, np.array(ele).reshape(1, -1), fmt = '%.6f', newline = '\n') for ele in rst]
    print('OutputData' + file_na[0] + '.txt' + ' saved in ' + PathOut + ' successfully!')