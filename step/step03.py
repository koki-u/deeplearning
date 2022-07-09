import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class  Function:
    def __call__(self, input):
        #箱の中のデータを取り出す
        x = input.data
        #実際の計算
        y = self.forward(x)
        #Variableの箱に入れ直す
        output = Variable(y)
        return output
#このforwardメソッドを使ってしまった人は、ミスっているから、今使っているメソッドの中でforwardの関数を書いてね。継承してねというサイン
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

#y = e^x　の計算を実装する
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

#y = (e^(x^2))^2　の計算を実装する
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

#こうすることで複雑な関数の計算を効率よく行うことができる。
#バックプロパゲーションの準備をしていた。