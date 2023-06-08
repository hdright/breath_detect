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

## 数据格式

### Cfg

这里指的是 `SampleSet.Cfg`

- `Nsamp`：样本数，`int`
- `Np`：各样本所含人数，`int[Nsamp]`
- `Ntx`：发射天线数，固定为1
- `Nrx`：接收天线数，`int`
- `Nsc`：子载波数，`int`
- `Nt`：各样本时域测量次数，`int[Nsamp]`
- `Tdur`：各样本信号采集持续时间（秒），`int[Nsamp]`
- `fstart`：第一个子载波中心频率（Mhz），`float`
- `fend`：最后一个子载波中心频率（Mhz），`float`

### CSI

这里指的是 `SampleSet.CSI_s`

为 `list`，由 `Nsamp` 个 `Nrx * Ntx * Nsc * Nt` 的矩阵构成

