'''
@file estBreath.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-06-08 13:40:24
@modified: 2023-06-08 19:02:29
'''

import numpy as np
from scipy.signal import find_peaks

def estChusai(Cfg: dict, CSI: np.ndarray, iSamp: int = 0) -> np.ndarray:
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    
    Np, Nrx, Ntx, Nsc, Nt, Tdur = Cfg["Np"][iSamp], Cfg["Nrx"], Cfg["Ntx"], Cfg["Nsc"], Cfg["Nt"][iSamp], Cfg["Tdur"][iSamp]

    fs = Nt / Tdur
    BPMresol = 0.1
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

    for rx in range(Nrx):
        for tx in range(Ntx):
            for sc in range(Nsc):
                H[rx][tx][sc] = Wdft @ CSI[rx][tx][sc]
                Hs += H[rx][tx][sc]

    filteredHs = np.zeros_like(Hs)
    ret = []
    BPMrange = [[10, 30], [6, 40], [0, 60]]
    BPMrangeI = 0
    while len(ret) < Np:
        minBPM, maxBPM = BPMrange[BPMrangeI]
        BPMrangeI += 1
        minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
        filteredHs[minIndex: maxIndex] = Hs[minIndex:maxIndex]
        idx, _ = find_peaks(np.abs(filteredHs), distance = 5)

        highestPeak = np.argsort(-np.abs(filteredHs[idx]))
        if (minIndex in idx[highestPeak][:Np] or maxIndex in idx[highestPeak][:Np]) and BPMrangeI < len(BPMrange):
            continue

        ret = np.asarray(idx[highestPeak][:Np] * BPMresol)
    return ret