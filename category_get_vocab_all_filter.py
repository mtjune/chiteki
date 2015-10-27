import pymysql
import yaml

import pandas as pd
import numpy as np
import six.moves.cPickle as pickle

import igo

DIC_DIR = "/home/yamajun/workspace/tmp/igo_ipadic"
tagger = igo.tagger.Tagger(DIC_DIR)
def igo_parse(text):
    words = tagger.parse(text)
    outputs = [word.surface for word in words if word.feature.split(',')[0] == '名詞']
    return outputs




if __name__ == '__main__':

    setting = None
    with open('mysql_setting.yml', 'r') as f:
        setting = yaml.load(f)


    connection = pymysql.connect(host=setting['host'],
                                 user=setting['user'],
                                 password=setting['password'],
                                 db='rakuten_recipe',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.SSCursor)


    FILTER_NUM = 9

    vocab_set = set()
    vocab_count = {}

    count_invalid = 0

    with connection.cursor() as cursor:
        sql = "select title from recipes;"

        cursor.execute(sql)

        count = 0
        for row in cursor:
            title = row[0]
            n_words = igo_parse(title)
            n_words_set = set(n_words)

            flag_invalid = True
            for n_word in n_words_set:
                if n_word not in vocab_set:
                    if n_word in vovab_count:
                        vocab_count[n_word] += 1
                        if vocab_count[n_word] > 4:
                            vocab_set.add(n_word)
                            del vocab_count[n_word]
                    else:
                        vocab_count = 1
                flag_invalid = False

            if flag_invalid:
                count_invalid += 1

            if count % 10000 == 0:
                print('end {} : vocab size : {}'.format(count, len(vocab_set)))
            count += 1


    print('end all : vocab size : {}'.format(len(vocab_set)))
    print('drop by filter word size : {}'.format(len(vocab_count)))

    vocab = {}
    count = 0
    for word in vocab_set:
        vocab[word] = count
        count += 1

    print('{} == {}'.format(len(vocab_set), len(vocab)))

    with open('result/category_vocab_all_filter_b.out', 'wb') as f:
        pickle.dump(vocab, f, -1)

    print('complete!')
