# encoding: utf-8

import pymysql
import yaml
import argparse

import pandas as pd
import numpy as np
import six.moves.cPickle as pickle



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default='result/vocab_1_b.out')
    parser.add_argument('--output', '-o', default='result/vocab_dic_1_b.out')
    args = parser.parse_args()

    vocab = None
    with open(args.data, 'rb') as f:
        vocab = pickle.load(f)


    vocab_dic = {word:i for i, word in enumerate(vocab)}

    print('process end: {0} == {1}'.format(len(vocab), len(vocab_dic)))

    with open(args.output, 'wb') as f:
        pickle.dump(vocab_dic, f, -1)


    print("complete!")
