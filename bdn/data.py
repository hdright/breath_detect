# %%
import numpy as np
from scipy.signal import savgol_filter
from itertools import accumulate
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale, StandardScaler
from chusai import CsiFormatConvrt, FindFiles, CfgFormat
import os
from skimage import transform

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
        if no_sample == 90:
            names = [self.names[0]]
        elif no_sample == 320:
            names = self.names[1:]
        elif no_sample == 90320: # 将90*600的fft矩阵扩展为320*600
            names = self.names
        else:
            names = []
            raise ValueError('no_sample should be 90 or 320')
        scaler = StandardScaler()
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

                CSIfft = np.zeros((Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'], dftSize), dtype=complex)
                for i in range(Cfg['Nrx']):
                    for j in range(Cfg['Ntx']):
                        for k in range(Cfg['Nsc']):
                            CSIfft[i * Cfg['Ntx'] * Cfg['Nsc'] + j * Cfg['Nsc'] + k] = np.fft.fft(
                                savgol_filter(np.abs(CSI_sam[i, j, k, :]), 8, 7), Ndft)[0:dftSize]
                CSIfft = np.abs(CSIfft)
                if no_sample == 90320 or no_sample == 90:
                    if Cfg['Nrx'] * Cfg['Ntx'] * Cfg['Nsc'] == 90:
                        # 将90*600的fft矩阵扩展为320*600
                        CSIfft_concat = np.zeros((320, noBpmPoints))
                        # 每30行拉伸为40行
                        for i in range(3):
                            CSIfft_concat[i*40:(i+1)*40, :] = transform.resize(CSIfft[i*30:(i+1)*30, :], (40, noBpmPoints), order=3)
                        # 再重复前120行至320行
                        CSIfft_concat[120:240, :] = CSIfft_concat[:120, :]
                        CSIfft_concat[240:, :] = CSIfft_concat[:80, :]
                        CSIfft = CSIfft_concat
                        
                # CSIfft按行标准化
                CSIfft = scaler.fit_transform(CSIfft.T).T
                self.CSI_fft.append(CSIfft.astype(np.float32))

                # 生成标签数据，是cfg['gt']的概率的类one-hot编码
                if self.round == 0:
                    gt_pd = np.zeros(noBpmPoints)
                    sigma2 = 1000 #/ BPMresol
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