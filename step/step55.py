if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#CNNのネットワーク構造
#勉強済みなので、飛ばす
#畳み込み層とプーリング層を作る

#フィルタとカーネルは同じ意味で捉える。
#パディング：畳み込み層のメインの処理を行う前に、入力データの周囲に固定のデータを埋める。サイズ調整
#ストライド：フィルタの移動する間隔のこと。

#出力サイズの計算方法

def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

# //　：割り算。割り切れないなら小数点以下切り捨て

#H:hight, W:wide
#input_size
H, W = 4, 4
#kernel_size
KH, KW = 3, 3
#stride Vertical and horizontal
SH, SW = 1, 1
#padding　Vertical and horizontal
PH, PW = 1, 1

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)