import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import warnings
from tqdm import tqdm
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', 500)

warnings.filterwarnings('ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_accuracy(y_pred, y_true):
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    return (y_pred == y_true).mean()


def l2_logistic_regression(x, y, lr, lbd, dev_x, dev_y):
    w = np.zeros(x.shape[1])
    epsilon = 1e-8
    min_loss = float('inf')
    best_train_acc = 0
    best_dev_acc = 0
    train_acc_history = []
    dev_acc_history = []
    for i in tqdm(range(10000)):
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


def l1_logistic_regression(x, y, lr, lbd, dev_x, dev_y):
    w = np.zeros(x.shape[1])
    epsilon = 1e-8
    min_loss = float('inf')
    best_train_acc = 0
    best_dev_acc = 0
    train_acc_history = []
    dev_acc_history = []
    for i in tqdm(range(10000)):
        gradient = ((x.multiply(y - sigmoid(x.dot(w)), axis=0)).mean())
        w = w + lr * gradient
        w[1:] = np.sign(w[1:]) * np.maximum(np.abs(w[1:]) - (lr * lbd), np.zeros(w[1:].shape))

        train_acc = cal_accuracy(sigmoid(x.dot(w)), y)
        dev_acc = cal_accuracy(sigmoid(dev_x.dot(w)), dev_y)
        train_acc_history.append(train_acc)
        dev_acc_history.append(dev_acc)

        best_train_acc = max(best_train_acc, train_acc)
        best_dev_acc = max(best_dev_acc, dev_acc)

        loss = ((-1 * y * np.log(sigmoid(x.dot(w)))) - ((np.ones(x.shape[0]) - y) * np.log(
            np.ones(x.shape[0]) - sigmoid(x.dot(w))))).mean() + lbd * np.sum(np.abs(w[1:]))

        # print("iter={}, loss={}, train_acc={}, dev_acc={}".format(i + 1, loss, train_acc, dev_acc))
        min_loss = min(min_loss, loss)
        if np.linalg.norm(gradient) <= epsilon:
            # print("lr={}, min_loss={}".format(lr, min_loss))
            break
    loss = ((-1 * y * np.log(sigmoid(x.dot(w)))) - ((np.ones(x.shape[0]) - y) * np.log(
        np.ones(x.shape[0]) - sigmoid(x.dot(w))))).mean() + lbd * np.sum(np.abs(w[1:]))
    train_acc = cal_accuracy(sigmoid(x.dot(w)), y)
    dev_acc = cal_accuracy(sigmoid(dev_x.dot(w)), dev_y)
    print("lr={}, lambda={}, min_loss={}, best_train_acc={}, best_dev_acc={}, loss={}, train_acc={}, dev_acc={}".format(
        lr, lbd, min_loss,
        best_train_acc, best_dev_acc,
        loss, train_acc, dev_acc))
    return w, train_acc_history, dev_acc_history


##############################################################################
# Feature Engineering

# Normalizing
train = pd.read_csv('../IA2-train.csv')
dev = pd.read_csv('../IA2-dev.csv')
test = pd.read_csv('../IA2-test-small-v2-X.csv')

X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
dev_X = dev.iloc[:, :-1]
dev_Y = dev.iloc[:, -1]

numerical_col = ['Age', 'Annual_Premium', 'Vintage']

for col in numerical_col:
    X[col + '_2'] = np.power(X[col], 2)
    dev_X[col + '_2'] = np.power(dev_X[col], 2)
    test[col + '_2'] = np.power(test[col], 2)

numerical_col = ['Age', 'Annual_Premium', 'Vintage', 'Age_2', 'Annual_Premium_2', 'Vintage_2']
numerical_col_mean = []
numerical_col_std = []

for col in numerical_col:
    numerical_col_mean.append(X[col].mean())
    numerical_col_std.append(X[col].std())
    X[col] = (X[col] - X[col].mean()) / X[col].std()

for i, col in enumerate(numerical_col):
    dev_X[col] = (dev_X[col] - numerical_col_mean[i]) / numerical_col_std[i]
    test[col] = (test[col] - numerical_col_mean[i]) / numerical_col_std[i]

# Select high relationship features

remain_col = [col for col in X.columns if (('Region_Code' not in col) & ('Policy_Sales_Channel' not in col))]

region_col = [col for col in X.columns if 'Region_Code' in col]
for col in region_col:
    r = ((train.loc[:, col] == 1) & (train.loc[:, 'Response'] == 1)).sum()
    if r >= 40:
        remain_col.append(col)
    # print("{}:{}".format(col, r))

policy_col = [col for col in X.columns if 'Policy_Sales_Channel' in col]

for col in policy_col:
    r = ((train.loc[:, col] == 1) & (train.loc[:, 'Response'] == 1)).sum()
    if r >= 100:
        remain_col.append(col)
    # print("{}:{}".format(col, r))
X = X.loc[:, remain_col]
dev_X = dev_X.loc[:, remain_col]
test = test.loc[:, remain_col]

#################################################################################################################
# create combine features
cate_col = [col for col in X.columns if col not in numerical_col and col != 'dummy']

for i, col in enumerate(cate_col):
    for j in range(i + 1, len(cate_col)):
        X[str(col) + '_' + str(cate_col[j]) + '_TT'] = np.where((X[col] == 1) & (X[cate_col[j]] == 1), 1, 0)
        X[str(col) + '_' + str(cate_col[j]) + '_TF'] = np.where((X[col] == 1) & (X[cate_col[j]] == 0), 1, 0)
        X[str(col) + '_' + str(cate_col[j]) + '_FT'] = np.where((X[col] == 0) & (X[cate_col[j]] == 1), 1, 0)
        X[str(col) + '_' + str(cate_col[j]) + '_FF'] = np.where((X[col] == 0) & (X[cate_col[j]] == 0), 1, 0)
        dev_X[str(col) + '_' + str(cate_col[j]) + '_TT'] = np.where((dev_X[col] == 1) & (dev_X[cate_col[j]] == 1), 1, 0)
        dev_X[str(col) + '_' + str(cate_col[j]) + '_TF'] = np.where((dev_X[col] == 1) & (dev_X[cate_col[j]] == 0), 1, 0)
        dev_X[str(col) + '_' + str(cate_col[j]) + '_FT'] = np.where((dev_X[col] == 0) & (dev_X[cate_col[j]] == 1), 1, 0)
        dev_X[str(col) + '_' + str(cate_col[j]) + '_FF'] = np.where((dev_X[col] == 0) & (dev_X[cate_col[j]] == 0), 1, 0)
        test[str(col) + '_' + str(cate_col[j]) + '_TT'] = np.where((test[col] == 1) & (test[cate_col[j]] == 1), 1, 0)
        test[str(col) + '_' + str(cate_col[j]) + '_TF'] = np.where((test[col] == 1) & (test[cate_col[j]] == 0), 1, 0)
        test[str(col) + '_' + str(cate_col[j]) + '_FT'] = np.where((test[col] == 0) & (test[cate_col[j]] == 1), 1, 0)
        test[str(col) + '_' + str(cate_col[j]) + '_FF'] = np.where((test[col] == 0) & (test[cate_col[j]] == 0), 1, 0)

cate_col = [col for col in X.columns if col not in numerical_col and col != 'dummy']
score = np.array([])

for col in cate_col:
    r = ((X.loc[:, col] == 1) & (train.loc[:, 'Response'] == 1)).sum()
    score = np.append(score, r)
    # print("{}:{}".format(col, r))

threshold = score.mean()
remain_col = []

for col in cate_col:
    r = ((X.loc[:, col] == 1) & (train.loc[:, 'Response'] == 1)).sum()
    if r >= threshold:
        remain_col.append(col)

X = X.loc[:, remain_col]
dev_X = dev_X.loc[:, remain_col]
test = test.loc[:, remain_col]

print(X.shape)

##################################################################################
# training model and save
weight_result, _, _ = l1_logistic_regression(X, Y, 0.01, 0.01, dev_X, dev_Y)

with open(r"l1_weight.pkl", "wb") as f:
    pickle.dump(weight_result, f)

weight_result, _, _ = l2_logistic_regression(X, Y, 0.01, 0.01, dev_X, dev_Y)

with open(r"l2_weight.pkl", "wb") as f:
    pickle.dump(weight_result, f)

with open(r"l1_weight.pkl", "rb") as f:
    l1_weight = pickle.load(f)

with open(r"l2_weight.pkl", "rb") as f:
    l2_weight = pickle.load(f)

prediction = pd.DataFrame(np.where(sigmoid(test.dot(l2_weight)) >= 0.5, 1, 0), columns=['Response'])
prediction.index.name = 'ID'
prediction.to_csv('submission.csv')
