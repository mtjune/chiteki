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

    category_recipe_ids = None
    with open('result/category_recipe_ids_b.out', 'rb') as f:
        category_recipe_ids = pickle.load(f)

    FILTER_NUM = 2

    vocab_set = set()
    vocab_count = {}

    for category, recipe_ids in category_recipe_ids.items():

        for recipe_id in recipe_ids:

            with connection.cursor() as cursor:
                sql = "select title from recipes where recipe_id = {};".format(recipe_id)

                cursor.execute(sql)

                result = cursor.fetchone()

                title = result[0]

                n_words = igo_parse(title)

                n_words_set = set(n_words)


                for n_word in n_words_set:
                    if n_word not in vocab_set:
                        if n_word in vocab_count:
                            vocab_count[n_word] += 1
                            if vocab_count[n_word] > 4:
                                vocab_set.add(n_word)
                                del vocab_count[n_word]
                        else:
                            vocab_count[n_word] = 1


        print(category, len(vocab_set))

    print('drop by filter word size : {}'.format(len(vocab_count)))

    vocab = {}
    count = 0
    for word in vocab_set:
        vocab[word] = count
        count += 1

    print('{} == {}'.format(len(vocab_set), len(vocab)))

    with open('result/category_vocab_filter_b.out', 'wb') as f:
        pickle.dump(vocab, f, -1)

    print('complete!')
