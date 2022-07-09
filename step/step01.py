#箱にデータを入れる。その箱に入った物を見ればデータがわかる。
class Variable:
    def __init__(self, data):
        self.data = data
        

import numpy as np
#xを見ることは箱を除く動作ということ。dataは1.0だから箱をみればdataがはいっている。
data = np.array(1.0)
x = Variable(data)
print(x.data)

#x.dataとかけば新しいデータが箱の中に入ってくるということ
x.data = np.array(2.0)
print(x.data)

#テンソルを使ってみる。
#0次元
x = np.array(1)
#.ndim　はnumber of dimensions　の意味で次元数を示している。
print(x.ndim)
#1次元
x = np.array([1, 2, 3])
print(x.ndim)
#2次元
x = np.array([[1, 2, 3],
            [4, 5, 6]])
print(x.ndim)

