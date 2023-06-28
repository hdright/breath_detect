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

# %%
PathSet = {0: "./TestData", 1: "./CompetitionData1", 2: "./CompetitionData2",
            3: "./CompetitionData3", 4: "./CompetitionData4"}
PrefixSet = {0: "Test", 1: "Round1", 2: "Round2", 3: "Round3", 4: "Round4"}

class BreathDataset(Dataset):
    def __init__(self, Round: int, 
                 path: str = "./chusai_data/", 
                 BPMresol: float = 1, 
                 breathEnd: int = 1, 
                 no_sample: int = 90):
        self.path = path
        self.round = Round
        if no_sample in [640, 180]:
            self.load_pha = True
        else:
            self.load_pha = False
        PathRaw = os.path.join(path, PathSet[Round])
        Prefix = PrefixSet[Round]

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
                names = [self.names[0]]
        elif no_sample % 320 == 0:
            if Round == 3 or Round == 2:
                names = self.names[1:3]
            else:
                names = self.names[1:9]
        elif no_sample == 90320: # 将90*600的fft矩阵扩展为320*600
            names = self.names
        else:
            names = []
            raise ValueError('no_sample should be 90 or 320')
        ampScalar = StandardScaler()
        phaScalar = StandardScaler()
        ampFftScaler = StandardScaler()
        phaFftScaler = StandardScaler()
        for na in names:  # 舍去第一个文件
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
                if self.round == 0:
                    with open(PathRaw + '/' + Prefix + 'GroundTruthData' + na + '.txt', 'r') as f:
                        gt = [np.fromstring(arr.strip(), dtype=float, sep = ' ') for arr in f.readlines()]
                    cfg['gt'] = np.pad(gt[ii], (0, 3 - len(gt[ii])), 'constant') # 补齐3个值, 防止dataloader因长度不一致报错
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

                csiAmpFft = np.zeros((Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'], dftSize), dtype=complex)
                if self.load_pha:
                    csiPhaFft = np.zeros((Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'], dftSize), dtype=complex)
                j = 0
                for i in range(Cfg['Nrx']):
                    for k in range(Cfg['Nsc']):
                        ampFiltered = savgol_filter(np.abs(CSI_sam[i, j, k, :]), 8, 7)
                        # ampFiltered = savgol_filter(hampelpd(np.abs(CSI_sam[i, j, k, :]), 400), 8, 7)
                        if no_sample % 90 != 0: # 3x30场景不scale时域幅度
                            if (j, k) == (0, 0):
                                ampScalar.fit(ampFiltered.reshape(-1, 1))
                            csiAmpScaled = ampScalar.transform(ampFiltered.reshape(-1, 1)).reshape(-1)
                            # csiAmpScaled = ampScalar.fit_transform(ampFiltered.reshape(-1, 1)).reshape(-1) # 不统一不对
                            csiAmpFft[i * Cfg['Nsc'] + k] = np.fft.fft(csiAmpScaled, Ndft)[0:dftSize]
                        else:
                            csiAmpFft[i * Cfg['Nsc'] + k] = np.fft.fft(ampFiltered, Ndft)[0:dftSize]
                        if self.load_pha:
                            phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]) -
                                                            np.angle(CSI_sam[(i+1)%Cfg['Nrx'], j, k, :]), 8, 7)
                            # if i>0:
                            #     phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]) -
                            #                                 np.angle(CSI_sam[(i+1)%Cfg['Nrx'], j, k, :]), 8, 7)
                            # else:
                            #     phaFiltered = savgol_filter(np.angle(CSI_sam[i, j, k, :]), 8, 7)
                            # 独立std
                            phaScaled = scale(phaFiltered)
                            csiPhaFft[i * Cfg['Nsc'] + k] = np.fft.fft(phaScaled, Ndft)[0:dftSize]
                csiAmpFft = np.abs(csiAmpFft)
                csiPhaFft = np.abs(csiPhaFft)
                # if no_sample == 90320 or no_sample == 90:
                if no_sample == 90320:
                    if Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'] == 90:
                        # 将90*600的fft矩阵扩展为320*600
                        CSIfft_stretch_sc = np.zeros((120, noBpmPoints))
                        CSIfft_stretch_rx = np.zeros((320, noBpmPoints))
                        # 每30行拉伸为40行
                        for i in range(3):
                            CSIfft_stretch_sc[i*40:(i+1)*40, :] = transform.resize(csiAmpFft[i*30:(i+1)*30, :], (40, noBpmPoints), order=3)
                        # 120行中，每隔40行取一行，一共能取3行（如0、40、80，等差数列），拉伸为8行，形成320行
                        for i in range(40):
                            sc_idx = np.arange(i, 120, 40)
                            rx_idx = np.arange(i, 320, 40)
                            CSIfft_stretch_rx[rx_idx, :] = transform.resize(CSIfft_stretch_sc[sc_idx, :], (8, noBpmPoints), order=3)
                        csiAmpFft = CSIfft_stretch_rx
                        
                # CSIfft按行标准化
                csiAmpFft = ampFftScaler.fit_transform(csiAmpFft.T).T
                csiPhaFft = phaFftScaler.fit_transform(csiPhaFft.T).T
                # # CSIfft按列标准化
                # csiAmpFft = ampFftScaler.fit_transform(csiAmpFft)
                # csiPhaFft = phaFftScaler.fit_transform(csiPhaFft)
                # 在现有的0axis之前创建一个新axis，并组合csiAmpFft和csiPhaFft
                csiAmpFft = np.expand_dims(csiAmpFft, axis=0)
                csiPhaFft = np.expand_dims(csiPhaFft, axis=0)
                csiFft = np.concatenate((csiAmpFft, csiPhaFft), axis=0)
                if self.load_pha:
                    self.CSI_fft.append(csiFft.astype(np.float32))
                else:
                    self.CSI_fft.append(csiAmpFft.astype(np.float32))

                # 生成标签数据，是cfg['gt']的概率的类one-hot编码
                if self.round == 0:
                    gt_pd = np.zeros(noBpmPoints)
                    sigma2 = (1 / BPMresol ** 2)
                    x = np.arange(noBpmPoints)
                    for p in range(cfg['Np']):
                        # gt所有值的高斯分布叠加作为label
                        mu = cfg['gt'][p] / BPMresol
                        gt_pd += np.exp(-(x - mu) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2) / cfg['Np']

                    self.gt_pd.append(gt_pd)

            del CSI
            

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
        BPMresolution = 1.0, # 设置BPM分辨率
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
    train_set = BreathDataset(Ridx, BPMresol=BPMresolution, no_sample=no_sample)
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