import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.initializers import HeNormal

class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(None, 2),
        )

    def __call__(self, x, y, train=True):
        return F.softmax_cross_entropy(self.fwd(x, train=train), y)

    def fwd(self, x, train=True):
        return self.l1(x)
