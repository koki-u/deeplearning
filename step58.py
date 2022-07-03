if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tkinter.tix import Tree
import numpy as np
from dezero.models import VGG16

model = VGG16(pretrained=True)

#ダミーのデータ
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
#model.plot(x)

import dezero
from PIL import Image

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/' \
    'raw/images/zebra.jpg'

img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
#img.show()
#しかし、PIL.Imageなので、ndarrayに変換

from dezero.models import VGG16

x = VGG16.preprocess(img)
print(type(x), x.shape)
#@statisticメソッドとしてpreprocessで変換できる

###################################################################
#分類をする

import numpy as np
from PIL import Image
import dezero.datasets
from dezero.models import VGG16

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/' \
    'raw/images/zebra.jpg'

img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img)
#バッチ用の軸を追加した
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

#計算グラフの可視化
model.plot(x, to_file='vgg.pdf')
#ImageNetのラベル
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
