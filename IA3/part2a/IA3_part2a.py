import numpy as np
import pandas as pd
import pickle
import warnings
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

train = pd.read_csv('../IA2-train.csv')
dev = pd.read_csv('../IA2-dev.csv')

# np.random.shuffle(train.values)
# np.random.shuffle(dev.values)

# normalization
numerical_col = ['Age', 'Annual_Premium', 'Vintage']
numerical_col_mean = []
numerical_col_std = []

for col in numerical_col:
    numerical_col_mean.append(train[col].mean())
    numerical_col_std.append(train[col].std())
    train[col] = (train[col] - train[col].mean()) / train[col].std()

dev = pd.read_csv('../IA2-dev.csv')

for i, col in enumerate(numerical_col):
    dev[col] = (dev[col] - numerical_col_mean[i]) / numerical_col_std[i]

# replace target 0 with -1
train['Response'][train['Response'] == 0] = -1
dev['Response'][dev['Response'] == 0] = -1


def predict_acc(x, y, w):
    return (np.where(x.dot(w) >= 0, 1, -1) == y).mean()


def kernel_function(x1, x2, p):
    return np.power(np.dot(x1, x2.T), p)


# print(train.iloc[0, :-1])
# print(kernel_function(dev.iloc[:, :-1], train.iloc[:, :-1], 1).shape)


def kernelized_perceptron(data, p, max_iter=100, training_size=6000, save=True):
    train, dev = data
    data = pd.concat([train, dev])
    x, y = data.iloc[:training_size, :-1], data.iloc[:training_size, -1]
    dev_x, dev_y = data.iloc[training_size:, :-1], data.iloc[training_size:, -1]

    start = time.time()

    k_train = kernel_function(x, x, p)
    k_valid = kernel_function(dev_x, x, p)
    a = np.zeros(x.shape[0])

    train_acc_his = []
    valid_acc_his = []

    for _ in tqdm(range(max_iter)):
        train_acc = 0
        valid_acc = 0
        for i in range(x.shape[0]):
            u = np.dot(k_train[i], a * y)
            if y.iloc[i] * u <= 0:
                a[i] = a[i] + 1
            else:
                train_acc += 1

        valid_acc = np.where(np.dot(k_valid, a * y) * dev_y > 0, 1, 0).sum()

        train_acc_his.append(train_acc / train.shape[0])
        valid_acc_his.append(valid_acc / dev.shape[0])

    end = time.time()
    if save:
        with open("train_acc_" + str(p) + ".pkl", 'wb') as f:
            pickle.dump(train_acc_his, f)

        with open("valid_acc_" + str(p) + ".pkl", 'wb') as f:
            pickle.dump(valid_acc_his, f)

    return end - start


######################################################################################
# (a)
for p in range(1, 6):
    kernelized_perceptron((train, dev), p)

    with open("train_acc_" + str(p) + ".pkl", 'rb') as f:
        train_acc_his = pickle.load(f)
    with open("valid_acc_" + str(p) + ".pkl", 'rb') as f:
        valid_acc_his = pickle.load(f)

    fig, ax = plt.subplots()
    ax.plot(train_acc_his, label="train accuracy")
    ax.plot(valid_acc_his, label="validation accuracy")
    plt.legend()
    plt.title("accuracy with p=" + str(p))
    plt.savefig("accuracy_p" + str(p) + ".png")

print(" ")

######################################################################################
# (b)
for p in range(1, 6):
    with open("train_acc_" + str(p) + ".pkl", 'rb') as f:
        train_acc_his = pickle.load(f)

    with open("valid_acc_" + str(p) + ".pkl", 'rb') as f:
        valid_acc_his = pickle.load(f)
    best_train_acc = np.argmax(np.array(train_acc_his))
    best_valid_acc = np.argmax(np.array(valid_acc_his))
    print("p={}".format(p))
    print("best train accuracy={} at iter={}".format(train_acc_his[best_train_acc], best_train_acc + 1))
    print("best validation accuracy={} at iter={}".format(valid_acc_his[best_valid_acc], best_valid_acc + 1))

print(" ")

######################################################################################
# (c)

run_time = {}

for ts in [10, 100, 1000, 10000]:
    run_time[str(ts)] = kernelized_perceptron((train, dev), 1, training_size=ts, save=False)

print(run_time)
fig, ax = plt.subplots()
ax.bar(run_time.keys(), run_time.values())
plt.title("run time respect to the size of training set")
plt.savefig("time.png")

print(" ")
