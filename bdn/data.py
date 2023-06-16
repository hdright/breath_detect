# %%
import numpy as np
from scipy.signal import savgol_filter
from itertools import accumulate
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale
from chusai import CsiFormatConvrt, FindFiles, CfgFormat
import os

# %%

class BreathDataset(Dataset):
    PathSet = {0: "./TestData", 1: "./CompetitionData1", 2: "./CompetitionData2",
               3: "./CompetitionData3", 4: "./CompetitionData4"}
    PrefixSet = {0: "Test", 1: "Round1", 2: "Round2", 3: "Round3", 4: "Round4"}

    def __init__(self, Round: int, path: str = "./chusai_data/", BPMresol: float = 1, breathEnd: int = 1):
        self.path = path
        self.round = Round
        PathRaw = os.path.join(path, self.PathSet[Round])
        Prefix = self.PrefixSet[Round]

        self.names = FindFiles(PathRaw)  # 查找文件夹中包含的所有比赛/测试数据文件
        dirs = os.listdir(PathRaw)
        names = []  # 文件编号
        files = []
        for f in sorted(dirs):
            if f.endswith('.txt'):
                files.append(f)
        for f in sorted(files):
            if f.find('CfgData') != -1 and f.endswith('.txt'):
                names.append(f.split('CfgData')[-1].split('.txt')[0])

        # self.CSI_s = []
        self.CSI_fft = []
        self.gt_pd = []
        self.Cfg = []
        for na in names[1:]:  # 舍去第一个文件
            print("%s / %d" % (na, len(names)))
            # 读取配置及CSI数据
            Cfg = CfgFormat(PathRaw + '/' + Prefix + 'CfgData' + na + '.txt')
            csi = np.genfromtxt(PathRaw + '/' + Prefix + 'InputData' + na + '.txt', dtype=float)
            CSI = csi[:, 0::2] + 1j * csi[:, 1::2]

            with open(PathRaw + '/' + Prefix + 'GroundTruthData' + na + '.txt', 'r') as f:
                 gt = [np.fromstring(arr.strip(), dtype=float, sep = ' ') for arr in f.readlines()]
            
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
                    'gt': gt[ii],
                }
                self.Cfg.append(cfg)
                # self.CSI_s.append(CsiFormatConvrt(CSI[Nt[ii]:Nt[ii+1], :], Cfg['Nrx'],
                #                                   Cfg['Ntx'], Cfg['Nsc'], Cfg['Nt'][ii]))
                CSI_sam = CsiFormatConvrt(CSI[Nt[ii]:Nt[ii+1], :], Cfg['Nrx'],
                                                  Cfg['Ntx'], Cfg['Nsc'], Cfg['Nt'][ii])
                fs = cfg['Nt'] / cfg['Tdur']  # 采样频率
                # BPMresol = 0.1
                resol = BPMresol / 60  # 要分辨出0.1BPM，需要的频率分辨率
                Ndft = int(fs / resol)  # DFT点数

                # breathEnd = 1  # 呼吸最高频率 Hz
                dftSize = int(breathEnd / resol)  # DFT宽度

                # 要估计的呼吸频率范围
                bpmRange = np.arange(0, 60, BPMresol)
                noBpmPoints = len(bpmRange)  # 要估计的呼吸频率个数

                CSIfft = np.zeros((Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'], dftSize), dtype=complex)
                for i in range(Cfg['Nrx']):
                    for j in range(Cfg['Ntx']):
                        for k in range(Cfg['Nsc']):
                            CSIfft[i * Cfg['Ntx'] * Cfg['Nsc'] + j * Cfg['Nsc'] + k] = np.fft.fft(
                                scale(savgol_filter(np.abs(CSI_sam[i, j, k, :]), 8, 7)), Ndft)[0:dftSize]
                self.CSI_fft.append(np.abs(CSIfft).astype(np.float32))

                # 生成标签数据，是cfg['gt']的概率的类one-hot编码
                gt_pd = np.zeros(noBpmPoints)
                sigma2 = 1 / BPMresol
                x = np.arange(noBpmPoints)
                for p in range(cfg['Np']):
                    # gt所有值的高斯分布叠加作为label
                    mu = cfg['gt'][p] / BPMresol
                    gt_pd += np.exp(-(x - mu) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2) / cfg['Np']

                self.gt_pd.append(gt_pd)

            del CSI
            

    def __len__(self) -> int:
        return len(self.CSI_fft)

    def __getitem__(self, idx) -> tuple[list, list, dict]:
        return self.CSI_fft[idx], self.gt_pd[idx], self.Cfg[idx]


def load_data_from_txt(
        Ridx = 0, # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
        BPMresolution = 1, # 设置BPM分辨率
        batch_size = 1, # 设置batch大小
        shuffle = True, # 设置是否打乱数据
        num_workers = 0, # 设置读取数据的线程数量
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
    train_set = BreathDataset(Ridx, BPMresol=BPMresolution)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader



