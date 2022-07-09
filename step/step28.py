if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph
import matplotlib as plt

#ローゼンブロック関数を実装する。
#バナナ関数とか呼ばれているらしい
def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

y = rosenbrock(x0 , x1)
y.backward()
#勾配ベクトルの話として計算できた。
print(x0.grad, x1.grad)

#勾配降下法の実装
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
#学修率
lr = 0.001
#繰り返す回数
iters = 1000

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

#グラフかきたいけどなんで無理？まあいいか
plt.plot(x0, x1)
plt.show()
