import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.initializers import HeNormal

from tqdm import tqdm
import argparse
import utils
from model import MyChain
from sys import exit
import pickle
from sklearn.externals import joblib

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='text', required=True)
    return parser.parse_args()

def evaluate(args):
    vectorizer = joblib.load('./vectorizer.pkl')
    with open('./m.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./eval.txt', 'r') as f:
        lines = f.readlines()
        lyrics = ' '.join(lines).replace('\n', '')
    vec = vectorizer.transform([lyrics]).toarray()
    x = Variable(np.array(vec).astype(np.float32))
    y = model.fwd(x, train=False)
    print(y.data)

if __name__ == '__main__':
    args = get_args()
    evaluate(args)
