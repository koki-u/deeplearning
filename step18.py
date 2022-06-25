import weakref
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
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


class Config:
    #Trueの時には、逆伝播を可能にする。有効モードとする。
    enable_backprop = True

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

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

#メモリの消費は抑えたい。
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
# t = x0 + x1 の計算後微分した数値に値を代入。
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)


#Config.enable_backprop の有効かどうかの切り替えを設定する。
Config.enable_backprop = True
x = Variable(np.array((100, 100, 100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprop = False
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))

#逆伝播のモードをきりかえる仕組みを行う。
#下のコードはcloseを毎回書かないといけないので、ややこしい。
"""
f = open('sample.txt', 'w')
f.write('hello world')
f.close()
"""

#勝手に閉じてくれるのがwithモードの利点
#これを利用して、逆伝播の無効モードに切り替える。
with open('sample.txt', 'w') as f:
    f.write('hello world')

#次のコードの中だけが、逆伝播の向こうモード
#もしこのコードから抜けると、逆伝播の有効モードに戻せる。
""""
with using_config("enable_backprop", False):
    x = Variable(np.array(2.0))
    y = square(x)
"""

#このwith文を使ったモード切り替えを実装する。
import contextlib

#デコレータをつけることで、文脈を判断する関数が作られる。
#この関数の中でyieldの前に前処理を書いて、yieldの後に後処理を書く。
#そうすると、with config_test():という構文を使うことができる。
#順番としては、前処理がよばれ→withの構文ブロック→後処理の順番
@contextlib.contextmanager
def config_test():
    #前処理
    print('start')
    try:
        yield
    finally:
        #後処理
        print('done')

with config_test():
    print('process...')


#nameのところにConfigの属性名を指定して、その指定されたnameをgetattr関数によってConfigクラスから取り出す。
#その後で,setattr関数によって新しい値を設定する。
#そうすることで、withブロックの中に入るときには、configクラスのnameで指定された属性がvalueに設定する。withブロックを抜けるときにold_valueに
#戻される。
import contextlib

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

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

#no_gradを呼ぶことで、順伝播だけで済む。
