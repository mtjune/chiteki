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


    genres = []
    try:
        with connection.cursor() as cursor:

            # ジャンル一覧を取得
            sql = "select id, name from genre;"
            cursor.execute(sql)
            for row in cursor:
                genres.append((row['id'], row['name']))



    finally:
        connection.close()

    print('length_n', len(genres))


    pickle.dump(genre_words, open('result/genres_b.out', 'wb'), -1)


    for g_id, g_name in genres:
        print(g_id, g_name)
