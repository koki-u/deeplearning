if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
#print(y)
#print(x.grad)

#numpyのnp.sumは和を求めるために軸も指定できる！！
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.sum(x, axis=0)
#print(y)
#print(x.shape, ' ->', y.shape)

#軸を指定した計算を行って生きたい！！

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)