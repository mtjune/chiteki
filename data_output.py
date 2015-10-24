# encoding: utf-8

import time
import argparse

import pandas as pd
import numpy as np
import six.moves.cPickle as pickle


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', '-d', default='result/genre_words.out')
    args = parser.parse_args()


    genres = None
    genre_words = None

    with open('result/genres_only_b.out', 'rb') as f:
        genres = pickle.load(f)

    with open(args.data, 'rb') as f:
        genre_words = pickle.load(f)



    for genre_id, words in sorted(genre_words.items(), key=lambda x:len(x[1]), reverse=True):
        genre_name = genres[genre_id]
        print(genre_name)
        for word, num in sorted(words.items(), key=lambda x:x[1], reverse=True)[:20]:
            print(' {0}\t{1}'.format(word, num))
