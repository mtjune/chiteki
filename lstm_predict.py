# -*- encoding: utf-8 -*-

import argparse
import math
import sys
import time
import yaml
import csv


import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import igo

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', '-c', default='result/vocab_dic_1_b.out')
parser.add_argument('--model', '-m', default='result/lstm_1.model.out')
parser.add_argument('--output', '-o', default='result/lstm')
parser.add_argument('--gpu', '-g', default=-1, type=int)
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

DIC_DIR = "/home/yamajun/workspace/tmp/igo_ipadic"
tagger = igo.tagger.Tagger(DIC_DIR)
def igo_parse(text):
    words = tagger.parse(text)
    outputs = [word.surface for word in words]
    return outputs


n_units = 650  # number of units per layer
batchsize = 1   # minibatch size

vocab = None
with open(args.vocab, 'rb') as f:
    vocab = pickle.load(f)

print('load vocab data')

def load_data(text):
    words = igo_parse(text)
    dataset = []
    for i, word in enumerate(words):
        if word not in vocab:
            print('{} not in vocab'.format(word))
            continue

        dataset.append(vocab[word])
    output = np.asarray(dataset, dtype=np.int32)
    return output


model = None
with open(args.model, 'rb') as f:
    model = pickle.load(f)

print('load model data')

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward_one_predict(x_data, state, train=False):
    # Neural net architecture
    x = chainer.Variable(x_data, volatile=not train)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(F.dropout(h2, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, F.softmax(y).data

def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units),
                                            dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

def predict(dataset, state=None):
    sum_log_perp = xp.zeros(())

    if state:
        in_state = state
    else:
        in_state = make_initial_state(batchsize=1, train=False)

    for i in range(len(dataset)):

        x_batch = xp.asarray(dataset[i])
        out_state, y = forward_one_predict(x_batch, in_state, train=False)
        in_state = out_state
        # out_text = [x[0] for x in vocab.items() if x[1] == y]
    return out_state, y


if __name__ == '__main__':
    print('input text')
    text = input()
    print('input', text)

    sentence = text

    state = None

    while(True):
        text_data = load_data(text)
        state, y = predict(text_data, state)
        y = y.reshape((-1,))
        ind = np.argmax(y)
        out_text = [x[0] for x in vocab.items() if x[1] == ind][0]
        sentence += out_text

        text = out_text

        if out_text == '。':
            print(sentence)
            sentence = ''
            if input() == 'end':
                break
