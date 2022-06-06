import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
class Node():
    attr_names = ("avg", "left", "right", "feature", "split", "mse")

    def __init__(self, avg=None, left=None, right=None, feature=None, split=None, mse=None):
        self.avg = avg
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.mse = mse

    def copy(self, node):
        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class MetaLearner(object):

    def __init__(self,min_samples,max_depth):
        self.root = Node()
        self.depth = 1
        self._rules = None
        self.min_samples=min_samples
        self.max_depth=max_depth

    @staticmethod
    def _expr2literal(expr: list) -> str:
        feature=expr
        operation=expr
        split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    def get_rules(self):
        que = [[self.root, []]]
        self._rules = []

        while que:
            node, exprs = que.pop(0)
            if not (node.left or node.right):
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.avg])
            if node.left:
                rule_left = np.copy(exprs)
                rule_left=np.append(rule_left,[node.feature, -1, node.split])
                que.append([node.left, rule_left])

            if node.right:
                rule_right = np.copy(exprs)
                rule_right = np.append(rule_right,[node.feature, 1, node.split])
                que.append([node.right, rule_right])

    @staticmethod
    def _get_split_mse(col: np.ndarray, label: np.ndarray, split: float) -> Node:
        label_left = label[col < split]
        label_right = label[col >= split]
        avg_left = label_left.mean()
        avg_right = label_right.mean()
        mse = (((label_left - avg_left) ** 2).sum() +
               ((label_right - avg_right) ** 2).sum()) / len(label)
        node = Node(split=split, mse=mse)
        node.left = Node(avg_left)
        node.right = Node(avg_right)

        return node

    def _choose_split(self, col: np.ndarray, label: np.ndarray) -> Node:
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node
        unique.remove(min(unique))
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(ite, key=lambda x: x.mse)
        return node

    def _choose_feature(self, data: np.ndarray, label: np.ndarray) -> Node:
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)
        node, feature = min(
            ite, key=lambda x: x[0].mse, default=(Node(), None))
        node.feature = feature

        return node

    def fit(self, data: np.ndarray, label: np.ndarray):

        self.root.avg = label.mean()
        que = [(self.depth + 1, self.root, data, label)]

        while que:
            depth, node, _data, _label = que.pop(0)
            if depth > self.max_depth:
                depth -= 1
                break
            if len(_label) < self.min_samples or all(_label == label[0]):
                continue
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue
            node.copy(_node)
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        self.depth = depth
        self.get_rules()

    def predict_one(self, row: np.ndarray) -> float:

        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
        return node.avg

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.predict_one, 1, data)

reg=MetaLearner(max_depth=20,min_samples=5)
class randomforest:
    def __init__(self, n_estimators, random_state):
        # 随机森林的大小
        self.n_estimators = n_estimators
        # 随机森林的随机种子
        self.random_state = random_state

    def fit(self, X, y):
        # 决策树数组
        dts = []
        for i in range(self.n_estimators):
            reg.fit(X,y)
            dts.append(reg)
        self.trees = dts



    def predict(self, X):
        # 预测结果
        ys = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            dt = self.trees[i]
            ys += dt.predict(X)
        ys /= self.n_estimators
        return ys

#划分数据集

data= load_diabetes()
x = data.data
y = data.target
data_train, data_val, score_train, score_val = train_test_split(x,y,test_size=0.2)
print(score_val)

#利用本模型预测
mse = []
for i in range(1,1000,10):
    r1 = randomforest(i,42)
    r1.fit(data_train,score_train)
    pre = r1.predict(data_val)
    print("本模型预测值：")
    print(pre)
    mse.append(np.sum((score_val-pre)**2) / len(score_val))
print("本模型误差")
print(mse)
#利用sklearn库预测
from sklearn.ensemble import RandomForestRegressor
mse1 = []  #存放误差
for i in range(1,1000,10):
    r1 = RandomForestRegressor(n_estimators=i)
    r1.fit(data_train,score_train)
    pre1 = r1.predict(data_val)
    print("sklearn预测值")
    print(pre1)
    mse1.append(np.sum((score_val-pre1)**2) / len(score_val))
print("sklearn误差")
print(mse1)

#画图
import matplotlib.pyplot as plt
n = [i for i in range(1,101)]
plt.plot(n,mse,color="red")
plt.plot(n,mse1,color="green")
plt.show()
