'''
@file estBreath.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-06-08 13:40:24
@modified: 2023-06-13 18:03:47
'''

import numpy as np
from robustica import RobustICA
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import pandas as pd

def hampelpd(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    # 如果是numpy数组，转换为pandas series
    if isinstance(vals_orig, np.ndarray):
        vals = pd.Series(vals_orig)
    else:
        #Make copy so original not edited
        vals = vals_orig.copy()
    
    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    
    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''
    
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    if isinstance(vals_orig, np.ndarray):
        return(vals.values)
    return(vals)

def hampel(X, k):
    """
    Apply Hampel filter to a given array X with window size k.
    The filter replaces outliers with the median value of the window.
    
    Parameters
    ----------
    X (numpy.ndarray): Input array to filter.
    k (int): Window size for the filter.
    
    Returns
    -------
    numpy.ndarray: Filtered array.
    """
    length = X.shape[0] - 1
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）
 
    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

# 注意，csi的维度为M*N*T,rx*sc*T
pi = np.pi
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

    fs = Nt / Tdur  # 采样频率
    BPMresol = 1
    resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
    Ndft = int(fs / resol)  # DFT点数

    breathEnd = 0.9  # 呼吸最高频率 Hz
    dftSize = int(breathEnd / resol)  # DFT宽度

    m, n = np.meshgrid(np.arange(Nt), np.arange(dftSize))
    Wdft = np.exp(-1j * 2 * np.pi * m * n / Ndft)  # dft 矩阵

    Hs = np.zeros((dftSize), dtype=complex)

    filteredHs = np.zeros_like(Hs)
    ret = np.array([0])
    BPMrange = [[14, 26], [9, 40], [5, 50]]
    if Np > 1:
        BPMrange = [[8, 32], [5, 50]]
    BPMrangeI = 0

    # 用取最大的三个峰值求加权平均
    freqs = []
    weights = []
    for rx in range(Nrx):
        for tx in range(Ntx):
            for sc in range(Nsc):
                csi_norm = np.angle(CSI[rx][tx][sc]) - np.angle(CSI[(rx + 1) % Nrx][tx][sc])
                denoised = savgol_filter(csi_norm, 8, 7)  # 用Savitzky-Golay滤波器去噪
                amp = scale(denoised)  # z-score 归一化
                
                psd = np.abs(Wdft @ amp)  # 用 DFT 变换到频域
                # pha = scale(np.angle(denoised))
                # n = len(psd)
                # autor = np.correlate(psd, psd, mode="full")[n-1:]
                for BPMrangeI in range(len(BPMrange)):
                    minBPM, maxBPM = BPMrange[BPMrangeI]
                    minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
                    filteredHs = np.zeros_like(Hs, dtype=np.float64)
                    filteredHs[minIndex: maxIndex] =psd[minIndex:maxIndex]
                    idx, _ = find_peaks(filteredHs, height=0.001, distance=3 / BPMresol)
                    if len(idx) < 1:
                        continue
                    highestPeak = np.argsort(-filteredHs[idx])
                    for i in range(min(len(idx), 4)):
                        if Np > i / 2:
                            freqs.append(idx[highestPeak[i]] * BPMresol)
                            weights.append(filteredHs[idx[highestPeak[i]]])
                    break
    # 用带权重的KMeans 聚类
    kmeans = KMeans(n_clusters=Np, tol=3.5, n_init="auto").fit(np.array([freqs]).T, sample_weight=weights)
    # 将频率按类内数量排序
    cnt = np.bincount(kmeans.labels_)
    freqs = kmeans.cluster_centers_[np.argsort(-cnt)]

    ret = np.sort(np.squeeze(freqs[:Np], axis=1))
    return ret

def estSani(Cfg: dict, CSI: np.ndarray, iSamp: int = 0) -> np.ndarray:
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    # np.save("CSI%d.npy" % iSamp, CSI)
    Np, Nrx, Ntx, Nsc, Nt, Tdur = Cfg["Np"][iSamp], Cfg["Nrx"], Cfg["Ntx"], Cfg["Nsc"], Cfg["Nt"][iSamp], Cfg["Tdur"][iSamp]

    fs = Nt / Tdur  # 采样频率
    BPMresol = 1
    resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
    Ndft = int(fs / resol)  # DFT点数

    breathEnd = 0.9  # 呼吸最高频率 Hz
    dftSize = int(breathEnd / resol)  # DFT宽度

    m, n = np.meshgrid(np.arange(Nt), np.arange(dftSize))
    Wdft = np.exp(-1j * 2 * np.pi * m * n / Ndft)  # dft 矩阵

    Hs = np.zeros((dftSize), dtype=complex)

    filteredHs = np.zeros_like(Hs)
    ret = np.array([0])
    BPMrange = [[14, 26], [9, 40], [5, 50]]
    if Np > 1:
        BPMrange = [[8, 32], [5, 50]]
    BPMrangeI = 0

    # 用取最大的三个峰值求加权平均
    freqs = []
    weights = []
    CSI_sani = CSI_sanitization(np.squeeze(CSI))
    for rx in range(Nrx):
        for tx in range(Ntx):
            for sc in range(Nsc):
                csi_norm = CSI_sani[rx][sc] - CSI_sani[(rx + 1) % Nrx][sc]
                denoised = savgol_filter(csi_norm, 8, 7)  # 用Savitzky-Golay滤波器去噪
                amp = scale(denoised)  # z-score 归一化
                
                psd = np.abs(Wdft @ amp)  # 用 DFT 变换到频域
                # pha = scale(np.angle(denoised))
                # n = len(psd)
                # autor = np.correlate(psd, psd, mode="full")[n-1:]
                for BPMrangeI in range(len(BPMrange)):
                    minBPM, maxBPM = BPMrange[BPMrangeI]
                    minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
                    filteredHs = np.zeros_like(Hs, dtype=np.float64)
                    filteredHs[minIndex: maxIndex] =psd[minIndex:maxIndex]
                    idx, _ = find_peaks(filteredHs, height=0.001, distance=3 / BPMresol)
                    if len(idx) < 1:
                        continue
                    highestPeak = np.argsort(-filteredHs[idx])
                    for i in range(min(len(idx), 4)):
                        if Np > i / 2:
                            freqs.append(idx[highestPeak[i]] * BPMresol)
                            weights.append(filteredHs[idx[highestPeak[i]]])
                    break
    # 用带权重的KMeans 聚类
    kmeans = KMeans(n_clusters=Np, tol=3.5, n_init="auto").fit(np.array([freqs]).T, sample_weight=weights)
    # 将频率按类内数量排序
    cnt = np.bincount(kmeans.labels_)
    freqs = kmeans.cluster_centers_[np.argsort(-cnt)]

    ret = np.sort(np.squeeze(freqs[:Np], axis=1))
    return ret

def calcRER(CSIf: np.ndarray, fHigh: float) -> float:
    '''
    计算CSI的RER(Respiration Energy Ratio)
    ref: MultiSense: Enabling Multi-person Respiration Sensing with Commodity WiFi

    :param CSIf: 1D 频域 CSI数据 [dftsize]，0Hz ~ fHigh Hz
    :return: RER值
    '''
    CSIpower = np.square(np.abs(CSIf))
    powerSum = np.sum(CSIpower)
    assert powerSum > 0
    BPMrange = [5, 50]
    breathPower = np.sum(CSIpower[int(BPMrange[0] / 60 / fHigh * len(CSIf))                         : int(BPMrange[1] / 60 / fHigh * len(CSIf))])
    RER = breathPower / powerSum
    return RER


def estMultisense(Cfg: dict, CSI: np.ndarray, iSamp: int = 0) -> np.ndarray:
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    # if iSamp > 37:
    #     np.save("CSI%d.npy" % iSamp, CSI)
    Np, Nrx, Ntx, Nsc, Nt, Tdur = Cfg["Np"][iSamp], Cfg["Nrx"], Cfg["Ntx"], Cfg["Nsc"], Cfg["Nt"][iSamp], Cfg["Tdur"][iSamp]

    fs = Nt / Tdur  # 采样频率
    BPMresol = .1
    resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
    Ndft = int(fs / resol)  # DFT点数

    breathEnd = 1  # 呼吸最高频率 Hz
    dftSize = int(breathEnd / resol)  # DFT宽度

    m, n = np.meshgrid(np.arange(Nt), np.arange(dftSize))
    Wdft = np.exp(-1j * 2 * np.pi * m * n / Ndft)  # dft 矩阵

    # 计算每个子载波的RER
    RER_sc_ = [0.0] * Nsc
    RER_ = np.empty((Nsc, Nrx * Ntx))
    for sc in range(Nsc):
        for rx in range(Nrx):
            for tx in range(Ntx):
                CSIf = Wdft @ np.abs(CSI[rx][tx][sc])
                RER_[sc][rx * Ntx + tx] = calcRER(CSIf, breathEnd)
                RER_sc_[sc] += RER_[sc][rx * Ntx + tx]
        RER_sc_[sc] /= Nrx * Ntx
    # 选取RER较大的子载波
    maxRER = np.max(RER_sc_)
    usefulSc = np.where(RER_sc_ >= maxRER * 0.6)[0]
    if len(usefulSc) < Nsc:
        print("%d / %d" % (len(usefulSc), Nsc))

    ret = np.zeros((Np))
    BPMrange = [[10, 30], [6, 40], [5, 50]]
    BPMrangeI = 0
    # # 用取最大的三个峰值求加权平均
    # Amp = [0] * Np
    # WeightedSum = [0] * Np
    rica = RobustICA(n_components=Np, whiten='arbitrary-variance', 
                     verbose=False)  # 'arbitrary-variance'

    patterns_ = []
    topCSIidxs_ = np.zeros((Nsc, Np), dtype=int)
    for sc in usefulSc:
        # 选取RER最大的前Np个CSI
        topCSIidx = np.argsort(-RER_[sc])[:Np]
        topCSIidxs_[sc] = topCSIidx
        topCSI = CSI[:, :, sc, :].reshape(Nrx * Ntx, Nt).transpose()[:, topCSIidx]
        # 用独立成分分析提取 Np 个复数呼吸信号
        # S, _ = rica.fit_transform(np.abs(topCSI))
        # S = S.transpose()
        Sreal, _ = rica.fit_transform(topCSI.real)
        Simag, _ = rica.fit_transform(topCSI.imag)
        # S = Sreal.transpose() + 1j * Simag.transpose()
        Sreal, Simag = Sreal.transpose(), Simag.transpose()

        if len(Sreal) < Np or len(Simag) < Np:
            # 如果分解出的独立成分不足Np个，就跳过这个子载波
            usefulSc = np.delete(usefulSc, np.where(usefulSc == sc))
            continue

        for i in range(Np):
            realSmooth = savgol_filter(Sreal[i], 8, 7)  # 用Savitzky-Golay滤波器平滑曲线
            imagSmooth = savgol_filter(Simag[i], 8, 7)
            pattern = np.empty_like(realSmooth)
            patternVar = 0
            for theta in np.linspace(0, 2 * np.pi, 100):
                pattern_ = np.cos(theta) * realSmooth + np.sin(theta) * imagSmooth
                patternVar_ = np.var(pattern_)
                if patternVar_ > patternVar:
                    patternVar = patternVar_
                    pattern = pattern_
            patterns_.append(pattern)

    # 如果len(patterns_) < Np，说明有些子载波没有分解出足够的独立成分，就放宽RER条件
    if len(patterns_) < 1:
        # 选取RER排名前40%的子载波
        RER_sc_a = np.array(RER_sc_)
        usefulSc = np.argsort(-RER_sc_a)[:int(Nsc * 0.4)]
        patterns_ = []
        topCSIidxs_ = np.zeros((Nsc, Np), dtype=int)
        for sc in usefulSc:
            # 选取RER最大的前Np个CSI
            topCSIidx = np.argsort(-RER_[sc])[:Np]
            topCSIidxs_[sc] = topCSIidx
            topCSI = CSI[:, :, sc, :].reshape(Nrx * Ntx, Nt).transpose()[:, topCSIidx]
            # 用独立成分分析提取 Np 个复数呼吸信号
            # S, _ = rica.fit_transform(np.abs(topCSI))
            # S = S.transpose()
            Sreal, _ = rica.fit_transform(topCSI.real)
            Simag, _ = rica.fit_transform(topCSI.imag)
            # S = Sreal.transpose() + 1j * Simag.transpose()
            Sreal, Simag = Sreal.transpose(), Simag.transpose()

            if len(Sreal) < Np or len(Simag) < Np:
                # 如果分解出的独立成分不足Np个，就跳过这个子载波
                usefulSc = np.delete(usefulSc, np.where(usefulSc == sc))
                continue

            for i in range(Np):
                realSmooth = savgol_filter(Sreal[i], 8, 7)  # 用Savitzky-Golay滤波器平滑曲线
                imagSmooth = savgol_filter(Simag[i], 8, 7)
                pattern = np.empty_like(realSmooth)
                patternVar = 0
                for theta in np.linspace(0, 2 * np.pi, 100):
                    pattern_ = np.cos(theta) * realSmooth + np.sin(theta) * imagSmooth
                    patternVar_ = np.var(pattern_)
                    if patternVar_ > patternVar:
                        patternVar = patternVar_
                        pattern = pattern_
                patterns_.append(pattern)
    # 用 K-Means 将呼吸模式聚类为 Np 类，对应 Np 个呼吸率
    kmeans = KMeans(n_clusters=Np).fit(patterns_)
    # 在每个类中，将所有呼吸模式按RER加权求和
    for i in range(Np):
        idxs_ = np.where(kmeans.labels_ == i)[0]
        # multiple sub-carriers combining
        r_msc = np.zeros((Nt))
        for idx in idxs_:
            r_msc += patterns_[idx] * calcRER(patterns_[idx], breathEnd)

        psd = Wdft @ r_msc  # 用 DFT 变换到频域
        # pha = scale(np.angle(denoised))
        for BPMrangeI in range(len(BPMrange)):
            minBPM, maxBPM = BPMrange[BPMrangeI]
            minIndex, maxIndex = int(minBPM / BPMresol), int(maxBPM / BPMresol)
            filteredHs = np.zeros_like(psd, dtype=np.float64)
            filteredHs[minIndex: maxIndex] = np.abs(psd[minIndex:maxIndex])
            idx, _ = find_peaks(filteredHs, height=0.001, distance=1)
            if len(idx) < 1:
                continue
            highestPeak = np.argsort(-filteredHs[idx])
            ret[i] = idx[highestPeak[0]] * BPMresol
            break
    ret = np.sort(ret)
    return ret

