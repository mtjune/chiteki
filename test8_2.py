# encoding: utf-8

# 市場データから，レビューよりどの要素がその商品の評価に関わるか

import pymysql
import yaml
import time

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
                                 cursorclass=pymysql.cursors.SSCursor)


    genres = []

    t0 = None
    t1 = None
    t2 = None
    try:
        with connection.cursor() as cursor:

            # ジャンル一覧を取得
            t0 = time.time()
            sql = "select id, name from genre;"
            cursor.execute(sql)

            t1 = time.time()

            keys = [x[0] for x in cursor.description]
            genres = [(row[keys.index('id')], row[keys.index('name')]) for row in cursor]

            t2 = time.time()

    finally:
        connection.close()

    print('length_n', len(genres))


    # pickle.dump(genres, open('result/genres_only_b.out', 'wb'), -1)


    print('1 - 0 : {}'.format(t1 - t0))
    print('2 - 1 : {}'.format(t2 - t1))


    for g_id, g_name in genres:
        print(g_id, g_name)
