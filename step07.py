import numpy as np

#バックプロパゲーションの自動化のために
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#逆向きに計算のグラフをたどる
#assert文は。assert ... の...がTrueでないと例外が発生する。
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)

#関数を取得
C = y.creator
#関数の入力を取得
b = C.input
#関数のbackwardメソッドを呼ぶ
b.grad = C.backward(y.grad)

B = b.creator
a = B.input
a.grad =  B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)

#これらはすべて同じことの繰り返しになっている。
#以下の文で自動化をする。

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    #再帰的な処理ができる。Noneになったら計算が終了する。
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)