if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero.datasets 

"""
x, t = dezero.datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])
"""

import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

#ハイパーパラメータを設定
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

#データの読み込み
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

#リストの初期化を行っておく。
trace_loss = []
trace_predict = []

for epoch in range(max_epoch):
    #データのインデックスのシャッフルをする
    index = np.random.permutation(data_size)
    sum_loss = 0
    
    for i in range(max_iter):
        #ミニバッチを作成
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        #勾配の算出とパラメータの更新
        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    #エポックごとに学修経過を出力する
    avg_loss = sum_loss / data_size

    trace_loss.append(avg_loss)
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

    # 作図
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(trace_loss)), trace_loss, label='train') # 損失
plt.xlabel('iterations (epoch)') # x軸ラベル
plt.ylabel('loss') # y軸ラベル
plt.title('Cross Entropy Loss', fontsize=20) # タイトル
plt.grid() # グリッド線
#plt.ylim(0, 0.2) # y軸の表示範囲
plt.show()

#lossが減っていくので、正しい方向に計算できているみたい。

# x軸の値を生成
x0_line = np.arange(-1.1, 1.1, 0.005)
print(x0_line[:5])

# y軸の値を生成
x1_line = np.arange(-1.1, 1.1, 0.005)
print(x1_line[:5])

# 格子状の点を生成
x0_grid, x1_grid = np.meshgrid(x0_line, x1_line)
print(x0_grid[:5, :5])
print(x0_grid.shape)
print(x1_grid[:5, :5])
print(x1_grid.shape)

# リストを結合
x_point = np.c_[x0_grid.ravel(), x1_grid.ravel()]
print(x_point)
print(x_point.shape)

# スコアを計算
y = model(x_point)
print(y.data)

# 推論結果(各データのクラス)を抽出
predict_cls = np.argmax(y.data, axis=1)
print(predict_cls)

# 形状を調整
y_grid = predict_cls.reshape(x0_grid.shape)
print(y_grid)

#各クラスのマーカーを指定しておく
markers = ['o', 'x', '^']
# 予測結果を作図
plt.figure(figsize=(8, 8))
plt.contourf(x0_grid, x1_grid, y_grid) # 予測クラス
for i in range(3):
    plt.scatter(x[t == i, 0], x[t == i, 1], marker=markers[i], 
                s=50, label='class ' + str(i)) # データセット
plt.xlabel('$x_0$', fontsize=15) # x軸ラベル
plt.ylabel('$x_1$', fontsize=15) # y軸ラベル
plt.suptitle('Spiral Dataset', fontsize=20) # 図全体のタイトル
plt.title('iter:' + str(max_epoch) + 
          ', loss=' + str(np.round(loss.data, 5)) + 
          ', N=' + str(len(x)), loc='left') # タイトル
plt.legend() # 凡例
plt.show()