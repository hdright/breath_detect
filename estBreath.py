'''
@file estBreath.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-06-08 13:40:24
@modified: 2023-06-09 11:22:00
'''

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import scale

def estChusai(Cfg: dict, CSI: np.ndarray, iSamp: int = 0) -> np.ndarray:
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    # np.save("CSI%d.npy" % iSamp, CSI)
    Np, Nrx, Ntx, Nsc, Nt, Tdur = Cfg["Np"][iSamp], Cfg["Nrx"], Cfg["Ntx"], Cfg["Nsc"], Cfg["Nt"][iSamp], Cfg["Tdur"][iSamp]

    fs = Nt / Tdur
    BPMresol = 1
    resol = BPMresol / 60 # 要分辨出0.1BPM，需要的频率分辨率
    Ndft = int(fs / resol) # DFT点数
    # print(Ndft)

    breathEnd = 0.7 # 呼吸最高频率
    dftSize = int(breathEnd / resol) # DFT宽度
    # print(dftSize)

    H = np.zeros((Nrx, Ntx, Nsc, dftSize), dtype = complex) # 用于存储DFT结果

    m, n = np.meshgrid(np.arange(Nt), np.arange(dftSize))
    Wdft = np.exp(-1j * 2 * np.pi * m * n / Ndft) # dft 矩阵

    Hs = np.zeros((dftSize), dtype = complex)


    filteredHs = np.zeros_like(Hs)
    ret = []
    BPMrange = [[10, 30], [6, 40], [5, 50]]
    BPMrangeI = 0
    if Np == 1:
        # 用取最大的三个峰值求加权平均
        Amp = 0
        WeightedSum = 0

        for rx in range(Nrx):
            for tx in range(Ntx):
                for sc in range(Nsc):
                    denoised = savgol_filter(np.abs(CSI[rx][tx][sc]), 8, 7) # 用Savitzky-Golay滤波器去噪
                    amp = scale(denoised) # z-score 归一化
                    psd = Wdft @ amp # 用 DFT 变换到频域
                    # pha = scale(np.angle(denoised))
                    for BPMrangeI in range(len(BPMrange)):
                        minBPM, maxBPM = BPMrange[BPMrangeI]
                        minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
                        filteredHs = np.zeros_like(Hs, dtype=np.float64)
                        filteredHs[minIndex: maxIndex] = np.abs(psd[minIndex:maxIndex])
                        idx, _ = find_peaks(filteredHs, height = 0.001, distance = 1)
                        if len(idx) < 1:
                            continue
                        highestPeak = np.argsort(-filteredHs[idx])
                        Amp += filteredHs[idx[highestPeak[0]]]
                        WeightedSum += filteredHs[idx[highestPeak[0]]] * idx[highestPeak[0]]
                        if len(idx) > 1:
                            Amp += filteredHs[idx[highestPeak[1]]]
                            WeightedSum += filteredHs[idx[highestPeak[1]]] * idx[highestPeak[1]]
                        if len(idx) > 2:
                            Amp += filteredHs[idx[highestPeak[2]]]
                            WeightedSum += filteredHs[idx[highestPeak[2]]] * idx[highestPeak[2]]
                        break

        ret = np.array([WeightedSum / Amp * BPMresol])
    else:
        for rx in range(Nrx):
            for tx in range(Ntx):
                for sc in range(Nsc):
                    denoised = savgol_filter(np.abs(CSI[rx][tx][sc]), 8, 7)
                    amp = scale(denoised) # z-score 归一化
                    psd = Wdft @ amp # 用 DFT 变换到频域
                    Hs += psd
        while len(ret) < Np:
            minBPM, maxBPM = BPMrange[BPMrangeI]
            BPMrangeI += 1
            minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
            filteredHs[minIndex: maxIndex] = Hs[minIndex:maxIndex]
            idx, _ = find_peaks(np.abs(filteredHs), distance = 5)

            highestPeak = np.argsort(-np.abs(filteredHs[idx]))
            if (minIndex in idx[highestPeak][:Np] or maxIndex in idx[highestPeak][:Np]) and BPMrangeI < len(BPMrange):
                continue

            ret = np.asarray(np.sort(idx[highestPeak][:Np]) * BPMresol)
    return ret
