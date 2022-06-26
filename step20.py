#Variableのところで演算も追加して行いたいって考える。
import weakref
import numpy as np
import contextlib

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    #再帰的な処理ができる。Noneになったら計算が終了する。
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            #outputsの微分をリストにまとめた。
            gys = [output().grad for output in f.outputs]
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
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    #yはweakref
                    #y()とすることによって、参照カウントを0にすることができる
                    y().grad = None

    def cleargrad(self):
        self.grad = None
    
    #この一行を入れることによって、shapeメソッドをインスタンス変数としてアクセスできるようにした。
    #x.shape()ではなくx.shapeとして扱えるということ。
    #shapeは何行何列なのか
    #ndimは次元数
    #dtypeはデータの型を表す。
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    #__len__は特殊メソッド
    #これで、len(x)を使うことで、Variableのndarrayの型でも使える。
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    #こうすることで、selfとotherが self * other としてmul関数に呼び出される。
    def __mul__(self, other):
        return mul(self, other)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        #(*xs)によって、self.forward(x0, x1)として呼び出せる。
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            #self.generationによって世代がうみだされ、それらを設定する。
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        #次のコードが順伝播の値をすべて保持しているところ。
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backforward(self, gy):
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Config:
    #Trueの時には、逆伝播を可能にする。有効モードとする。
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

#これで、逆伝播が不要なと金は、順伝播だけで済む。
#わざわざwith using_config。。。とかくと面倒なので、
#no_gradという関数を書く
def no_grad():
    return using_config('enable_backprop', False)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        #(y,) はタプルで返してほしいから
        return y
    
    #backwardは2つの変数が必要になる。
    #Variableクラスを修正しないといけない。
    def backward(self, gy):
        return gy, gy

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

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def add(x0, x1):
    return Add()(x0, x1)
Variable.__add__ = add

def square(x):
    
    f = Square()
    return f(x)

def mul(x0, x1):
    return Mul()(x0, x1)
Variable.__mul__ = mul

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

#y = add(mul(a, b), c)　のややこしい文章が、Variable.__mul__ = mul　によりmulを*だけで代用できるようになった。　
y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)
