#勾配降下法に変わるニュートン法を用いた最適化を行う！！
#勾配降下法だったら5万回しないと最小値は見つからなかったが、ニュートン法だと6回でもとまった
#ニュートン法はテイラー展開にて計算を行う。

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)

#これがニュートン法の威力！
#124回勾配降下法でやったけど、ニュートン法はわずか7回

#以下がplotしているもの
p = np.linspace(-3, 3, 50)
q = p ** 4 - 2 * p ** 2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 10)
plt.plot(p, q)
plt.show()

