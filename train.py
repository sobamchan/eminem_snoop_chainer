import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F

from tqdm import tqdm
import argparse
import utils
from model import MyChain
import pickle
from sklearn.externals import joblib

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', dest='epoch', type=int)
    parser.add_argument('--batch-size', dest='bs', type=int)
    return parser.parse_args()

def train(args):
    np.set_printoptions(threshold=np.nan)

    eminem_lyrics = utils.get_datas('./datas/eminem.json')
    snoop_lyrics = utils.get_datas('./datas/snoop_dogg.json')
    vectorizer = utils.get_vectorizer(eminem_lyrics+snoop_lyrics)

    x_train = np.array(eminem_lyrics[:int(len(eminem_lyrics)*0.8)] + snoop_lyrics[:int(len(snoop_lyrics)*0.8)])
    t_train = np.array([0] * int(len(eminem_lyrics)*0.8) + [1] * int(len(snoop_lyrics)*0.8)).astype(np.int32)
    x_test = np.array(eminem_lyrics[-int(len(eminem_lyrics)*0.2):] + snoop_lyrics[-int(len(snoop_lyrics)*0.2):])
    t_test = np.array([0] * int(len(eminem_lyrics)*0.2) + [1] * int(len(snoop_lyrics)*0.2)).astype(np.int32)

    epoch = args.epoch
    bs = args.bs
    train_N = len(x_train)
    test_N = len(x_test)

    model = MyChain()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    for j in tqdm(range(epoch)):
        sfindx = np.random.permutation(train_N)
        # train
        for i in range(0, train_N, bs):
            x_text = x_train[sfindx[i:i+bs]]
            x_vecs = vectorizer.transform(x_text).toarray()
            x_vecs = np.array(x_vecs).astype(np.float32)
            x = Variable(x_vecs)
            t = Variable(t_train[sfindx[i:i+bs]])
            model.cleargrads()
            loss = model(x, t)
            loss.backward()
            optimizer.update()

    # test
    x_text = x_test
    x_vecs = vectorizer.transform(x_text).toarray()
    x_vecs = np.array(x_vecs).astype(np.float32)
    x = Variable(x_vecs)
    t = Variable(t_test)
    y = model.fwd(x, train=False)
    acc = utils.accuracy(y.data, t.data)
    print('accuracy: {}%'.format(acc))
    with open('./m.pkl', 'wb') as f:
        pickle.dump(model, f)
    joblib.dump(vectorizer, './vectorizer.pkl')

    return acc

if __name__ == '__main__':
    args = get_args()
    acc = train(args)
