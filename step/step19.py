#使いやすくするために大量の名前について、変数をつけられるようにする。
#そのための name=None にする。

import weakref
import numpy as np

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


x = Variable(np.array([[1, 2, 3], [4 ,5, 6]]))
print(x.shape)
print(len(x))

#最後にVariableの中身を手軽に確認できる機能を追加する。
#printを使ってVariableの中のデータを出力する機能。

x = Variable(np.array([1, 2, 3]))
print(x)
#これを普通に出力しても、データの場所を教えてくれるだけになってしまう。
#def __repr__(self):　を書いたら、variable([1 2 3])と表示してくれる。

#以上で、Variableのはこの中身を透明にする。(普段どおりに使えるような実装をすることができた。)
