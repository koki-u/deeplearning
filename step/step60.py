if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print(sys.path)

#step59を改良する。
#1つめの改良は、時系列データようのデータローダを使って、複数データからなるミニバッチに対して順伝播
#2つめの改良は、RNNレイヤの代わりに、LSTMレイヤをつかう

#ミニバッチを使って学習するなら、1000個のデータに対し
#0-500, 501-1000 というようにスタート位置をずらす
from dezero.dataloaders import SeqDataLoader
import dezero.datasets
import numpy as np
import matplotlib.pyplot as plt
import dezero.optimizers
import dezero.functions as F
import dezero.layers as L
from dezero import Model

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)

class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        #hidden_sizeを利用した。
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

model = BetterRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1
    
        if count % bptt_length == 0 or count == seqlen:
            dezero.utils.plot_dot_graph(loss) #として計算グラフの可視化
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))


#めちゃくちゃ早くなった！！！！！

















#####################
#テスト用↓
"""
x, t = next(dataloader)
print(x)
print('-------------')
print(t)
"""