import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class  Function:
    def __call__(self, input):
        #箱の中のデータを取り出す
        x = input.data
        #実際の計算
        y = x**2
        #Variableの箱に入れ直す
        output = Variable(y)
        return output

x = Variable(np.array(10))
f = Function()
#__call__メソッドはpythonの特殊なメソッドで、このメソッドを定義すれば、f(x)のように書くことができる！凄い！！
y = f(x)

print(type(y))
print(y.data)

#DeZeroを使う時は、Functionクラスを基底クラスとして、すべての関数に共通する機能を実装する。
#具体的な関数は、Functionクラスを継承したクラスで実装する。

class Square(Function):
    def forward(self, x):
        return x ** 2

x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(x))
print(y.data)