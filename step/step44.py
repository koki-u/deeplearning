#IDK it doesn't work...

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero import Parameter
import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p
print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))

#isinstanceクラスでそれらを区別することができる！！
#これを利用して、Parameter インスタンスだけを集める仕組みを作ることができる。

#データセットを準備する。
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

#出力のサイズを指定できる
l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)