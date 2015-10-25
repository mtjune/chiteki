# encoding: utf-8

# 市場データから，レビューよりどの要素がその商品の評価に関わるか

import pymysql
import yaml

import argparse
import pandas as pd
import numpy as np
import six.moves.cPickle as pickle




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', '-d', default='result/word_scores_3_b.out')
    args = parser.parse_args()


    word_scores = None

    with open(args.data, 'rb') as f:
        word_scores = pickle.load(f)
    print('data loaded')

    word_posneg = {key:x[0] / x[1] for key, x in word_scores.items() if x[1] > 10}
    print('data computed')

    with open('result/word_posng_3_b.out', 'wb') as f:
        pickle.dump(word_posneg, f, -1)

    print('data saved!')
