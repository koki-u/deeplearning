import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    #再帰的な処理ができる。Noneになったら計算が終了する。
    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        #出力変数の生みの親を覚えさせる。
        output.set_creator(self)
        #入力された値を入れておく。
        self.input = input
        #出力も覚える
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backforward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
#backwardは関数の微分の値
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
#backwardは関数の微分の値
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

#これは2回書かないといけないのでめんどくさい。
x = Variable(np.array(0.5))
f = Square()
y = f(x)
#1つめの改良
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

#もしくは以下のように書くこともできる。
"""
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)
"""

x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

#逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)

#次のように書くこともできる
x = Variable(np.array(0.5))
#より自然なコードになった
y = square(exp(square(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)

#2つめの改良
#backwardメソッドの簡略化
#こうすることで、32ビットか64ビットか区別せずに微分ができる。
class Variable:
    def __init__(self, data):
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

x = Variable(np.array(0.5))
#より自然なコードになった
y = square(exp(square(x)))
#y.grad = np.array(1.0) 書かなくても良くなる！！
y.backward()
print(x.grad)

#改良の3つめ
#データの型を間違えないようにするために
#Variableのクラスはndarrayのみを扱える仕様になっているが、それが他の型だとだめなのでエラーメッセージを出す。
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

#as_arrayの関数によってスカラーなら次元を1次元に直してくれて便利な関数として使える。
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

#Functionクラスに追加する。
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        #出力変数の生みの親を覚えさせる。
        output.set_creator(self)
        #入力された値を入れておく。
        self.input = input
        #出力も覚える
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backforward(self, gy):
        raise NotImplementedError()
