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
            #outputsの微分をリストにまとめた。
            gys = [output.grad for output in f.outputs]
            #関数fの逆伝播を呼び出す。
            gxs = f.backward(*gys)
            #もしタプルじゃない場合の保険
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                #同じ変数を考える時の和の形が考えられていない。
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None

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
    
    #backwardは2つの変数が必要になる。
    #Variableクラスを修正しないといけない。
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
#backwardは関数の微分の値
#input[0]をしたことで、複数形にできた
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    f = Square()
    return f(x)

x = Variable(np.array(3.0))
y = add(x, x)
print('y', y.data)

y.backward()
print('x.grad', x.grad)
#この結果はおかしい。正しくは、2になるはずなのに、1となった。

#Variableのところを編集することで正しく変えられた。

#同じ変数を使って別の計算を行うときにはまた問題が生まれてくる。
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)

#これは、一つ前の計算結果が利用されて答えが違って出てくる。
x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)
#Variableクラスにcleargradを追加