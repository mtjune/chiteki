# encoding: utf-8

# 市場データから，レビューよりどの要素がその商品の評価に関わるか

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
    outputs = []

    for word in words:
        features = word.feature.split(",")
        outputs.append([word.surface] + features)

    return outputs



if __name__ == '__main__':

    setting = None
    with open('mysql_setting.yml', 'r') as f:
        setting = yaml.load(f)


    connection = pymysql.connect(host=setting['host'],
                                 user=setting['user'],
                                 password=setting['password'],
                                 db='rakuten_ichiba',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)


    count = 0
    genres = {}
    try:
        with connection.cursor() as cursor:
            sql = "select genre.name as g_name, review.description as desc from genre join review on genre.id = review.item_genre_id;"
            cursor.execute(sql)
            for row in cursor:
                count += 1
                i_name = row['g_name']
                if i_name in genres:
                    genres[g_name] = {}

                morphs = igo_parse(row['desc'])
                morphs_filtered = [x[7] for x in morphs if x[1] == "名詞" or x[1] == "動詞"]

                for morph in morphs_filtered:
                    if morph in genres[g_name]:
                        genres[g_name][morph] += 1
                    else:
                        genres[g_name][morph] = 1




    finally:
        connection.close()


    print('count:', count)
    print('length_n', len(genres))


    pickle.dump(genres, open('result/genres_b.out', 'wb'), -1)


    for key, value in sorted(genres.items(), key=lambda x:len(x[1]), reverse=True):
        print(key)
        for key1, value1 in sorted(value.items(), key=lambda x:x[1], reverse=True):
            print("-\t{0}\t{1}".format(key1, value1))
