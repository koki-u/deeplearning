if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
import dezero.datasets
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt
from dezero import DataLoader

#Datasetクラスによって、データセットの取り扱いが共通化された。
#Datasetのクラスには、前処理ができるようになった。
#DataLoaderクラスに、Datasetからミニバッチをつくれるようにした。

train_set = dezero.datasets.MNIST(train=True, transform=None)
test_set = dezero.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))

#データに対する前処理は明示的に何も行わない。
#訓練用のデータとテスト用のデータの長さを確認する。

#0番目を取り出した
x, t = train_set[0]
#データとラベルが対になったタプル型
#形状を表す数値が<class 'numpy.ndarray'> (1, 28, 28)これ。
print(type(x), x.shape)
#ラベルはなにか？
print(t)

plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
#plt.show()
print('label: ', t)

def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    #255.0で割ると規格化したことになる
    x /= 255.0
    return x

train_set = dezero.datasets.MNIST(train=True, transform=f)
test_set = dezero.datasets.MNIST(train=False, transform=f)

#NNするぞ！
max_epoch = 5
batch_size = 100
hidden_size = 1000
#lr = 1.0

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

#model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

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

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))

#テスト用のデータセットで85%の認識精度を達成した。

#モデルの改良
#ReLU関数を使う
#すると。。。これだけで、92%を達成した。

