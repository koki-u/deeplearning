if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#for文が幾重にも重なったコードになる。それはnumpyでfor文を使いまくることになり処理が遅くなってしまう。
#そこで、im2colをつかう。
#im2col：image to column;画像から列へ により、入力データを展開できる。

#イメージ的には、立体(3階テンソル)を平べったく直す(1次元の列)

import numpy as np
import dezero.functions as F
from dezero import Variable

#バッチサイズ：1, チャンネル数：3, 高さ：7, 幅：７
x1 = np.random.rand(1, 3, 7, 7)
#im2colの引数に注目
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

#10個のデータ
#バッチサイズ：1０, チャンネル数：3, 高さ：7, 幅：７
x2 = np.random.rand(10, 3, 7, 7)
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)


from dezero.utils import pair

print(pair(1))
print(pair((1, 2)))
#int と　(int, int)の両方の形式にたいして　、2つのようそを　持つタプルができる

# conv2d
N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)
x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()
print(y.shape)  # (1, 8, 15, 15)
print(x.grad.shape)  # (1, 5, 15, 15)