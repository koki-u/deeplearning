if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#RNNで過去の状態も見てみる。
import numpy as np
from dezero.utils import plot_dot_graph
import dezero.functions as F
import dezero.layers as L
from dezero import Model

""""
#隠れ層のサイズだけを指定
rnn = L.RNN(10)
#ダミーデータとして(1×1)のデータを使った
x = np.random.rand(1, 1)
h = rnn(x)
print(h.shape)
"""
#RNNレイヤの隠れ状態を出力へと変換するLinearレイヤを使う
#インスタンス変数のfcにLinearレイヤを追加する
#Linearレイヤは、RNNレイヤの隠れ状態を受け取って出力する
class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

"""
#ダミーの系列データ
seq_data = [np.random.randn(1, 1) for _ in range(1000)]
xs = seq_data[0: -1]
ts = seq_data[1:]

model = SimpleRNN(10, 1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)

    cnt += 1
    if cnt == 2:
        model.cleargrads()
        #各パラメータの勾配が求められる。
        #バックプロパゲーションBPTT
        loss.backward()
        break
"""
#BPTTはいくらでもグラフを生成するため、途中で打ち切るのが大事

#plot_dot_graph(loss, to_file='step59-02.png')


#################################################################
#sin波の予測
import numpy as np
import dezero.datasets
import matplotlib.pyplot as plt
import dezero.optimizers

train_set = dezero.datasets.SinCurve(train=True)
print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

"""
#図を描画
xs = [example[0] for example in train_set]
ts = [example[0] for example in train_set]
plt.plot(np.arange(len(xs)), xs, label='xs')
plt.plot(np.arange(len(ts)), ts, label='ts')
#plt.show()
"""

#1つめの要素が入力データで、もう一つは教師データ(ラベル)

#RNNでサイン波データセットを学習させる。

#ハイパーパラメータを設定
max_epoch = 100
hidden_size = 100
#BPTTの長さ
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizers = dezero.optimizers.Adam().setup(model)

#学習の開始
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        #形状を(1,1)に変換
        #dezeroの入力は2階テンソルか4階テンソルでないといけない。
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            #つながりを切る
            loss.unchain_backward()
            optimizers.update()

    #最初、countではなく、len(loss)　にしていたからlossが40とか20の範囲になっていて焦った
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
#モデルのリセット
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#バッチサイズを大きくする。→エポックの処理時間が短くて済む
#ミニバッチをまとめて処理できるようにする。
