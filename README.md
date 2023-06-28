# Breath_Detect

从无线信号中检测呼吸

路径结构
```
.
├── chusai_data
│   ├── 赛题任务书-基于无线感知信号的参数估计.pdf
│   ├── CompetitionData1 -> /data/chusai_data/CompetitionData1
│   ├── demo_code.py
│   └── TestData -> /data/chusai_data/TestData
└── README.md
```

呼吸 BPM 为 10\~40，即 *0.16~0.66 Hz*，不算婴儿则为 10\~30，即 *0.16~0.5 Hz*

题目约定 BPM $\in [5, 50]$，即 *0.08~0.83 Hz*

## 数据格式

### Cfg

这里指的是 `SampleSet.Cfg`

- `Nsamp`：样本数，`int`
- `Np`：各样本所含人数，`int[Nsamp]`，<=3
- `Ntx`：发射天线数，固定为1
- `Nrx`：接收天线数，`int`
- `Nsc`：子载波数，`int`
- `Nt`：各样本时域测量次数，`int[Nsamp]`
- `Tdur`：各样本信号采集持续时间（秒），`int[Nsamp]`，30~60
- `fstart`：第一个子载波中心频率（MHz），`float`
- `fend`：最后一个子载波中心频率（MHz），`float`

### CSI

这里指的是 `SampleSet.CSI_s`

为 `list`，由 `Nsamp` 个 `Nrx * Ntx * Nsc * Nt` 的矩阵构成

