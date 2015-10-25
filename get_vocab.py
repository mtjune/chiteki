# encoding: utf-8

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

    outputs = [word.surface for word in words]

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


    recipe_ids = None
    vocab = []

    try:
        with connection.cursor() as cursor:

            sql = "select recipe_id from recipes where small_category = 'チーズケーキ';"
            print("query :", sql)
            cursor.execute(sql)
            print("query complete")

            recipe_ids = [row[0] for row in cursor]
            recipe_num = len(recipe_ids)

        count = 0
        count_word = 0
        for recipe_id in recipe_ids:
            with connection.cursor() as cursor:

                sql = "select memo from steps where recipe_id = {};".format(recipe_id)
                print("query :", sql)
                cursor.execute(sql)
                print("query complete")

                for row in cursor:
                    words = igo_parse(row[0])

                    for word in words:
                        if word not in vocab:
                            vocab.append(word)

                    count_word += 1

            count += 1
            if count % 100 == 0:
                print(count, count_word, len(vocab))


    finally:
        connection.close()

    print('process end', len(vocab))

    with open('result/vocab_1_b.out', 'wb') as f:
        pickle.dump(vocab, f, -1)


    print("complete!")
