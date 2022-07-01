#分類問題にチャレンジ！
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import as_variable
from dezero import Variable
import dezero.functions as F

"""
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.get_item(x, 1)
print(y)

y.backward()
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
indices = np.array([0, 0, 0])
y = F.get_item(x, indices)
print(y)

Variable.__getitem__ = F.get_item

y = x[1]
print(y)
"""

from dezero.models import MLP

model = MLP((10 ,3))
###########################
#ソフトマックス関数によって確立としてみなして、合計１になるようにする。

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

x = np.array([[0.2, -0.4]])
y = model(x)
p = softmax1d(y)
print('y : ', y)
print('p : ', p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)