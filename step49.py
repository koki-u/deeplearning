if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
from pandas import array
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, train=True):
        self.train = train
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        #indexは整数だけに対応する。
        assert np.isscalar(index)
        if self.label is None:
            return self.data[index], None
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

#本来はデータセットに描くべきクラスを仮に書いている。
"""
class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
"""

import dezero.datasets

train_set = dezero.datasets.Spiral(train=True)
#train_set[0]としてデータにアクセスしている
#入力データとラベルがタプルで返されている。
print(train_set[0])
print(len(train_set))

#大きいデータセットの場合には？
#__getitem__と__len__の2つのメソッドがDeZeroである要件となる。
class BigData(Dataset):
    def __getitem__(index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x, t

    def __len__(self):
        return 1000000

#ミニバッチをデータセットの中から取り出す！
train_set = dezero.datasets.Spiral()

batch_index = [0, 1, 2]
batch = [train_set[i] for i in batch_index]
# batch = [(data_0, labbel_0), (data_1, label_1), (data_2, label_2)]
#batchから取り出した複数のデータがリストとして格納されている。
#ndarrayインスタンスに変換する必要がある
x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])

print(x.shape)
print(t.shape)

#これでbatchの各要素からデータを取り出して、ndarray インスタンスに連結できる。
#ニューラルネットワークに出力可能

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # Print loss every epoch
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))