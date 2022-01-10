"""
Make the imports of python packages needed
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


class Node():
    """
    Node of decision tree

    Parameters:
    -----------
    prediction: int
        Class prediction at this node
    feature: int
        Index of feature used for splitting on
    split: int
        Categorical value for the threshold to split on for the feature
    left_tree: Node
        Left subtree
    right_tree: Node
        Right subtree
    """

    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


###################


def entropy(target_col):
    """
    This function takes target_col, which is the data column containing the class labels, and returns H(Y).

    """

    ######################
    # Filling in this part#
    ######################
    target_uniq = np.unique(target_col)
    entropy = 0
    for y in target_uniq:
        Py = (target_col == y).mean()
        entropy += -Py * np.log2(Py)
    return entropy


###################

###################


def InfoGain(data, split_attribute_name, target_name="class"):
    """
    This function calculates the information gain of specified feature. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """

    ######################
    # Filling in this part#
    ######################
    feature_uniq = np.unique(data[split_attribute_name])
    Information_Gain = entropy(data[target_name])
    for x in feature_uniq:
        x_col = data.loc[data[split_attribute_name] == x, [split_attribute_name, target_name]]
        Information_Gain -= (x_col.shape[0] / data.shape[0]) * entropy(x_col[target_name])
    return Information_Gain


###################

###################


def DecisionTree(data, features, depth, maxdepth, target_attribute_name="class"):
    """
    This function takes following parameters:
    1. data = the data for which the decision tree building algorithm should be run --> In the first run this equals the total dataset

    2. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset once we have splitted on a feature

    3. target_attribute_name = the name of the target attribute
    4. depth = the current depth of the node in the tree --- this is needed to remember where you are in the overall tree
    5. maxdepth =  the stopping condition for growing the tree

    """
    # First of all, define the stopping criteria here are some, depending on how you implement your code, there maybe more corner cases to consider
    """
    1. If max depth is met, return a leaf node labeled with majority class, additionally
    2. If all target_values have the same value (pure), return a leaf node labeled with majority class 
    3. If the remaining feature space is empty, return a leaf node labeled with majority class
    """
    pred = np.bincount(data[target_attribute_name]).argmax()
    if depth == maxdepth or np.unique(data[target_attribute_name]).size == 1 or len(features) == 0:
        return Node(pred.item(), None, None, None, None)
    # If none of the above holds true, grow the tree!
    # First, select the feature which best splits the dataset
    max_gain_col = ''
    max_gain = float("-inf")
    for col in features:
        info_gain = InfoGain(data, col, target_attribute_name)
        if info_gain > max_gain:
            max_gain_col = col
            max_gain = info_gain
    # Once best split is decided, do the following:
    """
    1. create a node to store the selected feature 
    2. remove the selected feature from further consideration
    3. split the training data into the left and right branches and grow the left and right branch by making appropriate cursive calls
    4. return the completed node
    """

    # Show split feature each node and information gain.
    # print("Depth={}, the split feature is {}, information gain={}".format(depth, max_gain_col, max_gain))
    left = data.loc[data[max_gain_col] == 0]
    right = data.loc[data[max_gain_col] == 1]
    features.remove(max_gain_col)
    if left.shape[0] != 0:
        left_node = DecisionTree(left, features, depth + 1, maxdepth)
    else:
        left_node = Node(pred.item(), None, None, None, None)
    if right.shape[0] != 0:
        right_node = DecisionTree(right, features, depth + 1, maxdepth)
    else:
        right_node = Node(pred.item(), None, None, None, None)

    node = Node(pred, max_gain_col, 0, left_node, right_node)

    return node


###################

###################


def predict(example, tree, default=1):
    """
    This function handles making prediction for an example, takes two parameters:
    1. The example

    2. The tree, which is a node
    This needs to be done in a recursive manner. First check if the node is a leaf, if so, return the prediction of the node. Otherwise, send the example down the appropriate subbranch with recursive call.
    """

    node = tree
    while True:
        if not node.feature:
            return node.prediction
        # print(node.feature, example[node.feature], node.split, node.prediction)
        if example[node.feature] == node.split:
            node = node.left_tree
        else:
            node = node.right_tree


train = pd.read_csv("../mushroom-train.csv")
valid = pd.read_csv("../mushroom-val.csv")

print(train.shape)

"""
(a) The first 3 splits features
"""

DecisionTree(train, list(train.columns[:-1]), 0, 10)

"""
(b) plot accuracy of dmax from 1 to 10 
"""
train_acc = []
valid_acc = []
for d in range(10):
    DT = DecisionTree(train, list(train.columns[:-1]), 0, d + 1)

    train_pred = []
    for r in range(train.shape[0]):
        train_pred.append(predict(train.iloc[r, :], DT))
    train_acc.append((np.array(train_pred) == train.iloc[:, -1]).mean())

    valid_pred = []
    for r in range(valid.shape[0]):
        valid_pred.append(predict(valid.iloc[r, :], DT))
    valid_acc.append((np.array(valid_pred) == valid.iloc[:, -1]).mean())

    print("max depth={}, train acc={:.3f}, valid acc={:.3f}".format(d + 1, train_acc[d], valid_acc[d]))

with open(r"train_acc.pkl", "wb") as f:
    pickle.dump(train_acc, f)

with open(r"valid_acc.pkl", "wb") as f:
    pickle.dump(valid_acc, f)

with open(r"train_acc.pkl", "rb") as f:
    train_acc = pickle.load(f)

with open(r"valid_acc.pkl", "rb") as f:
    valid_acc = pickle.load(f)

fig, ax = plt.subplots()
ax.plot(train_acc, label="train accuracy", )
ax.plot(valid_acc, label="validation accuracy")
plt.legend()
plt.title("accuracy with respect to dmax from 1 to 10")
plt.savefig("accuracy.png")
