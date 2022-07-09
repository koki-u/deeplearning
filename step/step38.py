if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from venv import create
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

#要素ごとの計算を行わない関数について見ていく
#テンソルの形を整形するreshape機能
#転地をするためのtranspose関数

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

#これでVariableのすうちとして使えるようになった！！

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
print(y)
y = x.reshape(2, 3)
print(y)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(x.grad)

#transpose使って転置行列にしても大丈夫！

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.transpose()
y = x.T