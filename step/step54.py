if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#NNでは、過学習が問題
#原因として

#訓練データが少ない。→データの拡張
# モデルの表現力が高すぎる→Dropoutが良い。実践的によく使われている

#dropout：ニューロンをランダムに無効にしながら学習する方法

import numpy as np

#60%を無効にする
dropout_ratio = 0.6
x = np.ones(10)

mask = np.random.rand(10) > dropout_ratio
y = x * mask
print(y)

#機械学習ならアンサンブル学習が使われる
#アンサンブル学習：複数のモデルを個別に学習させ、推論時には、その複数の出力を平均する手法。
#NNでは、これにより認識精度が数%向上することが実験的に分かっている。

#学習時
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask
print(y)
#テスト時
scale = 1 - dropout_ratio
y = x * scale
print(y)

#通常のDropoutの話
#############################################################################
#inverted Dropoutの話
#Iverted Dropout：スケールを学習時に合わせる。テスト時にスケール合わせをしていたのが、通常のやつ
#メリットは、動的に変化させられるので、dropout_rationを固定しなくても良いということ
scale = 1 - dropout_ratio
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask / scale

#テスト時
y = x

from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x)

#学習時
y = F.dropout(x)
print(y)

#テスト時
with test_mode():
    y = F.dropout(x)
    print(y)
