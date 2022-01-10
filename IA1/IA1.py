import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

data = pd.read_csv('IA1_train.csv')

# print(data.shape)

# Part 0 (5 pts) : Data preprocessing.

data = data.iloc[:, 1:]
# print(data.shape)

# convert date to year, month, day
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data = data.iloc[:, 1:]

# print(data.shape)

# (a) Add dummy feature


data['dummy'] = np.ones(data.shape[0])
# print(data.shape)

# (b)

data.loc[data['yr_renovated'] == 0, 'age_since_renovated'] = data['year'] - data['yr_built']
data.loc[data['yr_renovated'] != 0, 'age_since_renovated'] = data['year'] - data['yr_renovated']

# (c)


col_mean = [0] * data.shape[1]
col_std = [0] * data.shape[1]

for i, col in enumerate(data.columns):
    if col == 'waterfront' or col == 'price' or col == 'dummy':
        continue
    col_mean[i] = data[col].mean()
    col_std[i] = data[col].std()
    data[col] = (data[col] - col_mean[i]) / col_std[i]


# Part 1 (40 pts). Implement batch gradient descent and explore different learning rates.


def batch_gradient(x, lr, y, epsilon, itr=float("inf")):
    w = np.zeros(x.shape[1])
    i = 0
    mse_arr = [np.power(x.dot(w) - y, 2).mean()]
    while i < itr:
        i += 1
        new_w = x.multiply((x.dot(w) - y), axis=0).mean() * 2

        w = w - lr * new_w
        mse = np.power(x.dot(w) - y, 2).mean()
        mse_arr.append(mse)

        if np.linalg.norm(new_w) <= epsilon or mse == np.nan or mse == float("inf"):
            break

    print("itr={}, lr={}, loss={}".format(i, lr, mse))
    return w, mse_arr


# loading dev data
dev = pd.read_csv('IA1_dev.csv')

# pre-processing dev data
dev = dev.iloc[:, 1:]

dev['date'] = pd.to_datetime(dev['date'])
dev['year'] = dev['date'].dt.year
dev['month'] = dev['date'].dt.month
dev['day'] = dev['date'].dt.day
dev = dev.iloc[:, 1:]

dev['dummy'] = np.ones(dev.shape[0])

dev.loc[dev['yr_renovated'] == 0, 'age_since_renovated'] = dev['year'] - dev['yr_built']
dev.loc[dev['yr_renovated'] != 0, 'age_since_renovated'] = dev['year'] - dev['yr_renovated']

for i, col in enumerate(dev.columns):
    if col == 'waterfront' or col == 'price' or col == 'dummy':
        continue
    dev[col] = (dev[col] - col_mean[i]) / col_std[i]


# compute_mse function
def compute_mse(x, y, w):
    return np.power(x.dot(w) - y, 2).mean()


# (a)

# lr_arr = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# mse_arr = []
# for lr in lr_arr:
#     w, mse = batch_gradient(data.drop('price', axis=1), lr, data['price'], 1e-8, itr=5000)
#     mse_arr.append(mse)
#
# fig = plt.figure()
# axe = fig.add_subplot()
# for i, lr in enumerate(lr_arr):
#     axe.plot(mse_arr[i], label=lr)
#
# plt.ylim(0, 100)
# plt.title("MSE with different lr")
# plt.xlabel("iterations")
# plt.ylabel("MSE")
# axe.legend()
# plt.savefig("MSE plot.png")
# plt.show()

# (b)

# lr_arr = [1e-1, 1e-2, 1e-3]
# weight_array = []
# for lr in lr_arr:
#     w, mse = batch_gradient(data.drop('price', axis=1), lr, data['price'], 1e-8, itr=5000)
#     weight_array.append(w)
#
# for i, w in enumerate(weight_array):
#     print("lr={}, MSE={}".format(lr_arr[i], compute_mse(dev.drop("price", axis=1), dev['price'], w)))

# (c)

# w, mse = batch_gradient(data.drop('price', axis=1), 0.1, data['price'], 1e-8, itr=5000)
# print("lr={}, MSE={}".format(0.1, compute_mse(dev.drop("price", axis=1), dev['price'], w)))
#
# print(w)
#
# print("The most positive weight feature is {} and The most negative weight feature is {}".format(
#     data.drop("price", axis=1).columns[np.argmax(w)], data.drop("price", axis=1).columns[np.argmin(w)]))

# # Part 2 (20 pts). Training with non-normalized data

# # reload data
# data = pd.read_csv('IA1_train.csv')
# # drop index
# data = data.iloc[:, 1:]
# # create date column
# data['date'] = pd.to_datetime(data['date'])
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day'] = data['date'].dt.day
# data = data.iloc[:, 1:]
# # adding dummy variable
# data['dummy'] = np.ones(data.shape[0])
# # create age_since_renovated feature
# data.loc[data['yr_renovated'] == 0, 'age_since_renovated'] = data['year'] - data['yr_built']
# data.loc[data['yr_renovated'] != 0, 'age_since_renovated'] = data['year'] - data['yr_renovated']
#
# lr_arr = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
# mse_arr = []
# for lr in lr_arr:
#     w, mse = batch_gradient(data.drop('price', axis=1), lr, data['price'], 1e-8, itr=5000)
#     mse_arr.append(mse)
#
# fig = plt.figure()
# axe = fig.add_subplot()
# for i, lr in enumerate(lr_arr):
#     axe.plot(mse_arr[i], label=lr)
#
# plt.ylim(0, 100)
# plt.title("MSE with different lr")
# plt.xlabel("iterations")
# plt.ylabel("MSE")
# axe.legend()
# plt.savefig("non-normalized MSE plot.png")
# plt.show()

# (c)

# # reloading dev data
# dev = pd.read_csv('IA1_dev.csv')
#
# # pre-processing dev data
# dev = dev.iloc[:, 1:]
#
# dev['date'] = pd.to_datetime(dev['date'])
# dev['year'] = dev['date'].dt.year
# dev['month'] = dev['date'].dt.month
# dev['day'] = dev['date'].dt.day
# dev = dev.iloc[:, 1:]
#
# dev['dummy'] = np.ones(dev.shape[0])
#
# dev.loc[dev['yr_renovated'] == 0, 'age_since_renovated'] = dev['year'] - dev['yr_built']
# dev.loc[dev['yr_renovated'] != 0, 'age_since_renovated'] = dev['year'] - dev['yr_renovated']
#
# w, mse = batch_gradient(data.drop('price', axis=1), 1e-11, data['price'], 1e-8, itr=5000)
#
# print("lr={}, MSE={}".format(1e-11, compute_mse(dev.drop("price", axis=1), dev['price'], w)))
#
# print(w)
#
# print("The most positive weight feature is {} and The most negative weight feature is {}".format(
#     data.drop("price", axis=1).columns[np.argmax(w)], data.drop("price", axis=1).columns[np.argmin(w)]))

# Part 3 (10 pts) Redundancy in features.


# (a)


# w, mse = batch_gradient(data.drop(['price', 'sqft_living15'], axis=1), 0.1, data['price'], 1e-8, itr=5000)
# print("lr={}, MSE={}".format(0.1, compute_mse(dev.drop(['price', 'sqft_living15'], axis=1), dev['price'], w)))
#
# print(w)

# Part 4 (15 pts). Explore feature-engineering and Participate in-class Kaggle competition.

# loading training data
train = pd.read_csv('PA1_train1.csv')
# one-hot-encoding zipcode
train_zipcode = pd.get_dummies(train['zipcode'])
train = train.drop(['id', 'sqft_living15', 'zipcode'], axis=1)

# The land space subtract living space
train['sqft_non_living'] = train['sqft_lot'] - train['sqft_living']
# The days since the building is built
train['date'] = pd.to_datetime(train['date'])
train.loc[train['yr_renovated'] == 0, 'days_since_built'] = (pd.Timestamp.now() - train['date']).dt.days
train.loc[train['yr_renovated'] != 0, 'days_since_built'] = (
        datetime.today() - pd.to_datetime(train['yr_renovated'])).dt.days

# the location of the house
lat_mean = train['lat'].mean()
long_mean = train['long'].mean()
train.loc[(train['lat'] > lat_mean) & (train['long'] > long_mean), 'ne'] = 1
train.loc[(train['lat'] <= lat_mean) & (train['long'] > long_mean), 'se'] = 1
train.loc[(train['lat'] > lat_mean) & (train['long'] <= long_mean), 'nw'] = 1
train.loc[(train['lat'] <= lat_mean) & (train['long'] <= long_mean), 'sw'] = 1
# is larger than neighbors
train.loc[train['sqft_lot'] > train['sqft_lot15'], 'is_larger_than_neighbors'] = 1

train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train = train.drop('date', axis=1)

train['dummy'] = np.ones(train.shape[0])

train.loc[train['yr_renovated'] == 0, 'age_since_renovated'] = train['year'] - train['yr_built']
train.loc[train['yr_renovated'] != 0, 'age_since_renovated'] = train['year'] - train['yr_renovated']

col_mean = [0] * train.shape[1]
col_std = [0] * train.shape[1]

train = train.fillna(0)

binary_col = ['waterfront', 'dummy', 'ne', 'se', 'nw', 'sw', 'is_larger_than_neighbors']
for i, col in enumerate(train.drop("price", axis=1).columns):
    if col in binary_col:
        continue
    col_mean[i] = train[col].mean()
    col_std[i] = train[col].std()
    train[col] = (train[col] - col_mean[i]) / col_std[i]

train = pd.concat([train, train_zipcode], axis=1)

# loading test data

submission = pd.read_csv('PA1_test1.csv')['id']

test = pd.read_csv('PA1_test1.csv')
# one-hot-encoding zipcode
test_zipcode = pd.get_dummies(test['zipcode'])
test = test.drop(['id', 'sqft_living15', 'zipcode'], axis=1)

# The land space subtract living space
test['sqft_non_living'] = test['sqft_lot'] - test['sqft_living']
# The days since the building is built
test['date'] = pd.to_datetime(test['date'])
test.loc[test['yr_renovated'] == 0, 'days_since_built'] = (datetime.today() - test['date']).dt.days
test.loc[test['yr_renovated'] != 0, 'days_since_built'] = (
        datetime.today() - pd.to_datetime(test['yr_renovated'])).dt.days
# the position of the house
test.loc[(test['lat'] > lat_mean) & (test['long'] > long_mean), 'ne'] = 1
test.loc[(test['lat'] <= lat_mean) & (test['long'] > long_mean), 'se'] = 1
test.loc[(test['lat'] > lat_mean) & (test['long'] <= long_mean), 'nw'] = 1
test.loc[(test['lat'] <= lat_mean) & (test['long'] <= long_mean), 'sw'] = 1
# is larger than neighbors
test.loc[test['sqft_lot'] > test['sqft_lot15'], 'is_larger_than_neighbors'] = 1

test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test = test.drop('date', axis=1)

test['dummy'] = np.ones(test.shape[0])

test.loc[test['yr_renovated'] == 0, 'age_since_renovated'] = test['year'] - test['yr_built']
test.loc[test['yr_renovated'] != 0, 'age_since_renovated'] = test['year'] - test['yr_renovated']

test = test.fillna(0)

for i, col in enumerate(test.columns):
    if col in binary_col:
        continue
    test[col] = (test[col] - col_mean[i]) / col_std[i]

test = pd.concat([test, test_zipcode], axis=1)

print(train.shape)
print(test.shape)

# original model
w, _ = batch_gradient(train.drop('price', axis=1), 0.1, train['price'], 1e-8, itr=20000)

predict_price = test.dot(w)

submission = pd.DataFrame({'id': submission, 'price': predict_price})
submission.to_csv('submission.csv', index=False)
