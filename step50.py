if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

#DataLoaderクラスを作る
#ミニバッチの作成やデータセットのシャッフルを行なうクラス

#まずは、イテレータ：反復子として順番にデータを取り出す機能
#関数も存在するが、自作することも可能

class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    #特殊メソッド__iter__
    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()

        self.cnt += 1
        return self.cnt

obj = MyIterator(5)
for x in obj:
    print(x)

#確かに取り出せることが確認できた
#このように与えられたデータセットを先頭から順に取り出して、必要に応じてデータセットのシャッフルを行なう

from dezero.datasets import Spiral
from dezero import DataLoader

batch_size = 10
max_epoch = 1

#訓練用のDataLoaderは、エポック毎にデータシャッフルをおこなうので、True
train_set = Spiral(train=True)
#精度評価ように用いるだけなので、False
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break

    for x, t in test_loader:
        print(x.shape, t.shape)
        break

import numpy as np
import dezero.functions as F

y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
t = np.array([1, 2, 0])
acc = F.accuracy(y, t)
print("認識精度：{}".format(acc))

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

loss_list = []
accuracy_list = []


train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        #訓練用のミニバッチ
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        #訓練データの認識精度
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    #勾配が不要な場合のモード
    with dezero.no_grad():
        #訓練用のミニバッチデータ
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            #テストデータの認識精度
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    loss_list.append(sum_loss / len(train_set))
    accuracy_list.append(sum_loss / len(train_set))
