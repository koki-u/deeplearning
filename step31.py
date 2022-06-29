if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

#これも解説で終わっちゃったyo!
#とりあえずgx.backward()をつくっちゃえ。そうすると高階微分になるヨンってことね