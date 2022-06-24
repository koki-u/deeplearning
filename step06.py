import numpy as np

#gradのインスタンスは微分用のもの。Noneで初期化しておく。逆伝播のときにその値を設定する。
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        #入力された値を入れておく。
        self.input = input
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)

#A→B→Cの順に微分をもとめたのでは無く。C→B→Aで求めた。