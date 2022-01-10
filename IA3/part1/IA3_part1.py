import numpy as np
import pandas as pd
import pickle
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

train = pd.read_csv('../IA2-train.csv')
dev = pd.read_csv('../IA2-dev.csv')

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


online_train_acc = []
online_dev_acc = []
average_train_acc = []
average_dev_acc = []


def average_perceptron(data, max_iter=100):
    train, dev = data
    x, y = train.iloc[:, :-1], train.iloc[:, -1]
    dev_x, dev_y = dev.iloc[:, :-1], dev.iloc[:, -1]
    online_w = np.zeros(x.shape[1])
    avg_w = np.zeros(x.shape[1])
    s = 1

    for _ in tqdm(range(max_iter)):
        for i in range(x.shape[0]):
            xi = x.iloc[i, :]
            yi = y[i]
            if xi.dot(online_w) * yi <= 0:
                online_w = online_w + (yi * xi)
            avg_w = ((s * avg_w) + online_w) / (s + 1)
            s += 1
        online_train_acc.append(predict_acc(x, y, online_w))
        online_dev_acc.append(predict_acc(dev_x, dev_y, online_w))
        average_train_acc.append(predict_acc(x, y, avg_w))
        average_dev_acc.append(predict_acc(dev_x, dev_y, avg_w))

    with open(r"online_train_acc.pkl", "wb") as f:
        pickle.dump(online_train_acc, f)

    with open(r"online_dev_acc.pkl", "wb") as f:
        pickle.dump(online_dev_acc, f)

    with open(r"average_train_acc.pkl", "wb") as f:
        pickle.dump(average_train_acc, f)

    with open(r"average_dev_acc.pkl", "wb") as f:
        pickle.dump(average_dev_acc, f)

    with open(r"weight.pkl", "wb") as f:
        pickle.dump((online_w, avg_w), f)

    return online_w, avg_w


# online_weight, average_weight = average_perceptron((train, dev))

fig, ax = plt.subplots()
acc_list = ['online_train_acc', 'online_dev_acc', 'average_train_acc', 'average_dev_acc']
for i in range(4):
    with open(acc_list[i] + ".pkl", "rb") as f:
        acc = pickle.load(f)
    ax.plot(acc, label=acc_list[i])
plt.legend()
plt.title("accuracy with various data and method")
plt.savefig('accuracy.png')

###################################################################################################
# part1 (b)
with open(r"online_dev_acc.pkl", "rb") as f:
    online_dev_acc = np.array(pickle.load(f))
with open(r"average_dev_acc.pkl", "rb") as f:
    average_dev_acc = np.array(pickle.load(f))

max_online_iter = np.argmax(online_dev_acc)
max_average_iter = np.argmax(average_dev_acc)

print("The best accuracy of online perceptron on validation set is {}, happened on iteration={}".format(
    online_dev_acc[max_online_iter], max_online_iter))
print("The best accuracy of average perceptron on validation set is {}, happened on iteration={}".format(
    online_dev_acc[max_average_iter], max_average_iter))
