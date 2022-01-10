import numpy as np
import pandas as pd
import pickle
import warnings
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

train = pd.read_csv('../IA2-train.csv')
# print(train.shape)
numerical_col = ['Age', 'Annual_Premium', 'Vintage']
numerical_col_mean = []
numerical_col_std = []

for col in numerical_col:
    numerical_col_mean.append(train[col].mean())
    numerical_col_std.append(train[col].std())
    train[col] = (train[col] - train[col].mean()) / train[col].std()

X = train.iloc[:, :-1]
Y = train.iloc[:, -1]

dev = pd.read_csv('../IA2-dev.csv')

for i, col in enumerate(numerical_col):
    dev[col] = (dev[col] - numerical_col_mean[i]) / numerical_col_std[i]

dev_X = dev.iloc[:, :-1]
dev_Y = dev.iloc[:, -1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_accuracy(y_pred, y_true):
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    return (y_pred == y_true).mean()


def logistic_regression(x, y, lr, lbd, dev_x, dev_y):
    w = np.zeros(x.shape[1])
    epsilon = 1e-8
    min_loss = float('inf')
    best_train_acc = 0
    best_dev_acc = 0
    train_acc_history = []
    dev_acc_history = []
    for i in range(10000):
        gradient = ((x.multiply(y - sigmoid(x.dot(w)), axis=0)).mean())
        w = w + lr * gradient
        w[1:] = w[1:] - lr * lbd * w[1:]

        train_acc = cal_accuracy(sigmoid(x.dot(w)), y)
        dev_acc = cal_accuracy(sigmoid(dev_x.dot(w)), dev_y)
        train_acc_history.append(train_acc)
        dev_acc_history.append(dev_acc)

        best_train_acc = max(best_train_acc, train_acc)
        best_dev_acc = max(best_dev_acc, dev_acc)

        loss = ((-1 * y * np.log(sigmoid(x.dot(w)))) - ((np.ones(x.shape[0]) - y) * np.log(
            np.ones(x.shape[0]) - sigmoid(x.dot(w))))).mean() + lbd * np.sum(np.power(w[1:], 2))

        # print("iter={}, loss={}, train_acc={}, dev_acc={}".format(i + 1, loss, train_acc, dev_acc))
        min_loss = min(min_loss, loss)
        if np.linalg.norm(gradient) <= epsilon:
            # print("lr={}, min_loss={}".format(lr, min_loss))
            break
    loss = ((-1 * y * np.log(sigmoid(x.dot(w)))) - ((np.ones(x.shape[0]) - y) * np.log(
        np.ones(x.shape[0]) - sigmoid(x.dot(w))))).mean() + lbd * np.sum(np.power(w[1:], 2))
    train_acc = cal_accuracy(sigmoid(x.dot(w)), y)
    dev_acc = cal_accuracy(sigmoid(dev_x.dot(w)), dev_y)
    print("lr={}, lambda={}, min_loss={}, best_train_acc={}, best_dev_acc={}, loss={}, train_acc={}, dev_acc={}".format(
        lr, lbd, min_loss,
        best_train_acc, best_dev_acc,
        loss, train_acc, dev_acc))
    return w, train_acc_history, dev_acc_history


# Run learning rate experiment with lambda=10

# for lr in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#     weight = logistic_regression(X, Y, lr, 10, dev_X, dev_Y)

#############################################################
# 1.a
# Do lambda test and save

train_acc_result = defaultdict(list)
dev_acc_result = defaultdict(list)
weight_result = defaultdict(list)

for lbd in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]:
    w, train_acc_array, dev_acc_array = logistic_regression(X, Y, 0.01, lbd, dev_X, dev_Y)
    train_acc_result[lbd] = train_acc_array
    dev_acc_result[lbd] = dev_acc_array
    weight_result[lbd] = w

with open(r"train_acc.pkl", "wb") as f:
    pickle.dump(train_acc_result, f)

with open(r"dev_acc.pkl", "wb") as f:
    pickle.dump(dev_acc_result, f)

with open(r"weight.pkl", "wb") as f:
    pickle.dump(weight_result, f)

# read lambda test record and plot it
with open(r"train_acc.pkl", "rb") as f:
    train_acc_result = pickle.load(f)

fig, ax = plt.subplots()
for key in train_acc_result.keys():
    plt.plot(train_acc_result[key], label="\u03BB={}".format(key))

plt.legend()
plt.title('train accuracy plot with different lambda')
plt.savefig("l2_train_acc.png")

with open(r"dev_acc.pkl", "rb") as f:
    dev_acc_result = pickle.load(f)

fig, ax = plt.subplots()
for key in dev_acc_result.keys():
    plt.plot(dev_acc_result[key], label="\u03BB={}".format(key))

plt.legend()
plt.title('validation accuracy plot with different lambda')
plt.savefig("l2_dev_acc.png")

##############################################################################
# 1.b
# show top 5 magnitude features of three lambda

top_5_weight = defaultdict(lambda: defaultdict(list))

with open(r"weight.pkl", "rb") as f:
    weight_result = pickle.load(f)

for lbd in [1e-3, 1e-2, 1e-1]:
    w = weight_result[lbd]
    for i in np.argpartition(-1 * np.abs(w), 5)[:5]:
        top_5_weight[lbd][X.columns[i]] = w[i]

print("")

for k in top_5_weight.keys():
    print("The top 5 features of \u03BB={} are {}".format(k, dict(top_5_weight[k])))

###############################################################################
# 1.c
# Show sparsity of the model as the number of weights that equal zero

with open(r"weight.pkl", "rb") as f:
    weight_result = pickle.load(f)

zero_count = []
bar_x = []
for key in weight_result.keys():
    zero_count.append(np.count_nonzero(weight_result[key] == 0))
    bar_x.append('\u03BB=' + str(key))

fig, ax = plt.subplots()
ax.bar(bar_x, zero_count)
plt.title('Sparsity for different \u03BB')
plt.savefig('l2_sparsity.png')
