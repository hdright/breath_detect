'''
@file estBreath.py
@author Sciroccogti (scirocco_gti@yeah.net)
@brief 
@date 2023-06-08 13:40:24
@modified: 2023-06-08 13:41:24
'''

import numpy as np

def estChusai(Cfg, CSI, iSamp = 0):
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据 [NRx][NTx][NSc][NT]
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    #########以下代码，参赛者用自己代码替代################
    #########样例代码中直接返回随机数作为估计结果##########
    result = np.random.rand(Cfg['Np'][iSamp]) * 45 + 5
    result = np.sort(result)
    return result