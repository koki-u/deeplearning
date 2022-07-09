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

class Exp(Function):

    def forward(self, x):
        return np.exp(x)

#数値微分に代わるより効率的なアルゴリズムのバックプロパゲーションを実装するのが目標
#まずは普通の微分から

#誤差が一番小さい方法　{f(x+h)+f(x-h)}/2h の傾きを求める
#eps はめちゃくちゃ小さい数字という意味
def numerical__diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data -y0.data) / (2 * eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical__diff(f, x)
print(dy)

#合成関数の微分を行う
#y = (e^(x^2))^2
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.abs(0.5))
dy = numerical__diff(f, x)
print(dy)

#これで自動的に関数の微分をもとめる事ができるようになったが、数値微分には限界が来る。
#桁落ちと,DNNではパラメータ多すぎてむずい
#バックプロパゲーションの登場
