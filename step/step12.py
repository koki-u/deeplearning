import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    #再帰的な処理ができる。Noneになったら計算が終了する。
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
#使う人にとっての改善
#def __call__(self, *inputs): にするリストを使わずに何でも任意の数を入れられる。
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        #出力変数の生みの親を覚えさせる。
        for output in outputs:
            output.set_creator(self)
        #入力された値を入れておく。
        self.inputs = inputs
        #出力も覚える
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backforward(self, gy):
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

#これまでは、1つの引数につき1つの戻り値を返していたが、それを複数で可能にした。
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        #(y,) はタプルで返してほしいから
        return (y,)



x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
f = Add()
y = f(x0, x1)
print(y.data)

#関数を実装しやすくする
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        #(*xs)によって、self.forward(x0, x1)として呼び出せる。
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        #出力変数の生みの親を覚えさせる。
        for output in outputs:
            output.set_creator(self)
        #入力された値を入れておく。
        self.inputs = inputs
        #出力も覚える
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backforward(self, gy):
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        #(y,) はタプルで返してほしいから
        return y

def add(x0, x1):
    return Add()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)

#可変長の順伝播ができるようになった。