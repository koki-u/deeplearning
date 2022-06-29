if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

def f(x):
    y = x ** 4 - 2 * x **2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
gx.backward()
print(x.grad)

#よっしできた！！
#微分が残った状態て新たに逆伝播を行ってしまったために、新しい微分値が加算されてしまった。

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)

#x.cleargrad()で呼び出したことにより一回微分がクリアされる！！

#ニュートン法による最適化
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data

#すばら。。ニュートン法の威力っすね！！
