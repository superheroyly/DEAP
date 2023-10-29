# Leaky integrator model of Echo State Network
import numpy as np
import time
import openpyxl
import generate_networks
import random
from sklearn.preprocessing import normalize
import send_mail as sm
import joblib
import os
# 划分训练集，验证集，测试集
# path = "D:\\Pycharm\\Project\\ML\\Dataset\\DEAP\\data_preprocessed_python\\DEAP41_signal0.5s\\"  # 32W 8W 每个subject按4：1划分
path = "D:\\Pycharm\\Project\\ML\\Dataset\\DEAP\\data_preprocessed_python\\DEAP\\"  # 32W 8W 每个subject按4：1划分


# Train set
with open(path + 'data.npy', 'rb') as fileTrain:
    data_train = np.load(fileTrain)
with open(path + 'label.npy', 'rb') as fileTrainL:
    label_train = np.load(fileTrainL)
data_train = normalize(data_train)
# data_train = data_train[:32000, :]
# label_train = np.ravel(label_train[:, ])  # 多维转一维 标签：Valence Arousal Domain Like

LA, HA, LV, HV, S, C = 0, 0, 0, 0, 0, 0

for i in label_train:
    if i[0] <= 5:
        LV += 1
    else:
        HV += 1

    if i[1] <= 5:
        LA += 1
    else:
        HA += 1

    if i[0] <= 3 and i[1] >= 5:
        S += 1
    elif i[0] >= 4 and i[0] <= 6 and i[1] < 4:
        C += 1

print("LV:{0}, HV:{1}".format(LV, HV))
print("LA:{0}, HA:{1}".format(LA, HA))
print("S:{0}, C:{1}".format(S, C))
print('Trainset and Testset OK!')
'''
LV:34320, HV:42480
LA:32580, HA:44220
S:7980, C:8760
'''