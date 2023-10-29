# Leaky integrator model of Echo State Network
import numpy as np
import time
import pickle
import torch
import generate_networks
import argparse
from sklearn.preprocessing import normalize
import openpyxl
import pandas as pd

is_cuda = torch.cuda.is_available()
print("Is_GPU:", is_cuda)

# 划分训练集，验证集，测试集
# path = "D:\\Pycharm\\Project\\ML\\Dataset\\DEAP\\data_preprocessed_python\\out51\\"
# path = "D:\\Pycharm\\Project\\ML\\Dataset\\DEAP\\data_preprocessed_python\\DEAP41\\"
path = "D:\\Pycharm\\Project\\ML\Dataset\\DEAP\\Human-Emotion-Analysis-using-EEG-from-DEAP-dataset\\psd analysis knn and svm\\"

# Trainset X Y Z
# with open(path + 'data_training.npy', 'rb') as fileTrain:
#     X = np.load(fileTrain)
# with open(path + 'label_training.npy', 'rb') as fileTrainL:
#     Y = np.load(fileTrainL)


# for label in Y:
#     if label[1] >= 5 and label[0] <= 3:
#         X1.append(X[i, :].tolist())
#         Y1.append(Y[i, :2].tolist())
#     if label[0] >= 4 and label[0] <= 6 and label[1] < 4:
#         X1.append(X[i, :].tolist())
#         Y1.append(Y[i, :2].tolist())
#     i += 1

# stress = 0
# clam = 1
# Arousal_Train = np.ravel(Y[:, [0]])
# Valence_Train = np.ravel(Y[:, [1]])
# Domain_Train = np.ravel(Y[:, [2]])
# Like_Train = np.ravel(Y[:, [3]])

data_train = []
data_test = []
label_test = []
label_train = []

for i in range(1, 32):
    if i < 10:
        b = '0' + str(i)
    else:
        b = str(i)
    f = 'testfile_' + b + '.txt'
    df = pd.read_csv(path+f)
    df.drop(['user'], 1, inplace=True)
    df.drop(['video'], 1, inplace=True)
    df.drop(['wavesegment'], 1, inplace=True)
    df.drop(['combined'], 1, inplace=True)
    df.drop(['arousal'], 1, inplace=True)

    X = np.array(df.drop(['valence'], 1))
    y = np.array(df['valence'])



# store_path = "D:\\Pycharm\\Project\\ML\\Dataset\\DEAP\\data_preprocessed_python\\DEAP41_4classify\\"
store_path = "D:\\Pycharm\\Project\\ML\Dataset\\DEAP\\Human-Emotion-Analysis-using-EEG-from-DEAP-dataset\\psd analysis knn and svm\\"


np.save(store_path+"data_train", np.array(data_train), allow_pickle=True, fix_imports=True)
np.save(store_path+"label_train", np.array(label_train), allow_pickle=True, fix_imports=True)
print("Training dataset:", np.array(data_train).shape, np.array(label_train).shape)

np.save(store_path+"data_test", np.array(data_test), allow_pickle=True, fix_imports=True)
np.save(store_path+"label_test", np.array(label_test), allow_pickle=True, fix_imports=True)
print("Testing dataset:", np.array(data_test).shape, np.array(label_test).shape)



M1 = []
N1 = []

# Testset M L
with open(path + 'data_testing.npy', 'rb') as fileTest:
    M = np.load(fileTest)
with open(path + 'label_testing.npy', 'rb') as fileTestL:
    N = np.load(fileTestL)

i = 0
for label in N:
    if label[1] >= 5 and label[0] <= 3:
        M1.append(M[i, :].tolist())
        N1.append(N[i, :2].tolist())
    if label[0] >= 4 and label[0] <= 6 and label[1] < 4:
        M1.append(M[i, :].tolist())
        N1.append(N[i, :2].tolist())
    i += 1

np.save(store_path+"data_test", np.array(M1), allow_pickle=True, fix_imports=True)
np.save(store_path+"label_test", np.array(N1), allow_pickle=True, fix_imports=True)
print("Testing dataset:", np.array(M1).shape, np.array(N1).shape)

print("OK！")

'''
4C training dataset: (71977, 70) (71977, 2)
4C testing dataset: (14399, 70) (14399, 2)
'''
'''
4:1数据集　不重叠窗口　所有通道　４个频段
Training dataset: (13392, 128) (13392, 2)
Testing dataset: (3348, 128) (3348, 2)
'''
