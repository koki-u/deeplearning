#layerを上手く作っていく
#まとめていく。今の時点だと、10層作るなら10層のlinearインスタンスを作らないといけない

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.metrics import max_error
import dezero.layers as L
import dezero.functions as F
from dezero import Variable
from dezero import Model
from dezero import Layer


model = Layer()
#出力サイズだけを指定する
model.l1 = L.Linear(5)
model.l2 = L.Linear(10)


#推論をおこなう関数
def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid_simple(y)
    y = model.l2(y)
    return y

#すべてのパラメータにアクセス
for p in model.params():
    print(p)

#すべてのパラメータの勾配をリセット
model.cleargrads()

#↑これでNNで使用するパラメータをまとめて管理できる。

#Layerクラスの継承をして、1つのクラスとしてモデルを定義する。

class TwoLayerNet(Model):
    #必要なlayerを生成して、l1として設定する
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    #推論を行うコードを書いていく。
    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y

x = Variable(np.random.randn(5, 10), name ='x')
model  = TwoLayerNet(100, 10)
model.plot(x)




###############################################################################
#パラメータの管理は、すべてModelクラスから管理する。汎用性のあるNNを作っていく。

#Modelを使って問題を解く
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

#ハイパーパラメータの設定t
lr = 0.2
max_iter = 10000
hidden_size = 10

#モデルの定義
class TwoLayerNetwork(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y 

model = TwoLayerNet(hidden_size, 1)

#学修の開始
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

#########################################
#全結合層ネットワークを作る。

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid_simple):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)

#2層のMLP
model = MLP((10, 1))
#5層のMLP
model = MLP((10, 20 ,30, 40, 1))
#説明
#初期化で、fc(=full connect)をタプルもしくはリストで指定する。
#activationは活性化関数の指定を行う。デフォルトはsigmoid_simple