#モジュール
#pythonファイルのことを言う。特に、他のpythonのプログラムからインポートして利用することを想定して作られたpythonファイルをモジュールと呼ぶ

#パッケージ
#パッケージは複数のモジュールをまとめたもの。パッケージを作るためには、ディレクトリを作り、その中にモジュールを追加する。

#ライブラリ
#複数のパッケージをまとめたもの。ファイル構成でいえば、1つ以上のディレクトリによって構成せれる。ただし、パッケージを指してライブラリと呼ぶこともある。
import sys
print(sys.path)
sys.path.append('DeepLearning_03')


import numpy as np
from dezero.core_simple import Variable

x = Variable(np.array(1.0))
print(x)

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)