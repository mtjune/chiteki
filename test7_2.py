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
    genre_words = {}
    try:
        with connection.cursor() as cursor:

            # ジャンル一覧を取得
            sql = "select id, name from genre;"
            cursor.execute(sql)
            keys = [x[0] for x in cursor.description]
            genres = [(row[keys.index('id')], row[keys.index('name')]) for row in cursor]

        print('get genres.')

        # ジャンル毎のレビュー取得 & 集計
        for genre_id, genre_name in genres:
            if genre_name in ["その他", "パソコン周辺機器"]:
                continue

            with connection.cursor() as cursor:
                t0 = time.time()
                if not genre_name in genre_words:
                    genre_words[genre_name] = {}

                sql = "select `description` from `review` where `item_genre_id` = %s limit 100;"
                print('query', genre_id, genre_name)
                cursor.execute(sql, (str(genre_id),))
                t1 = time.time()
                print('queried', genre_id, genre_name, t1 - t0)
                keys = [x[0] for x in cursor.description]
                desc_key = keys.index('description')
                t2 = time.time()
                print('get keys', genre_id, genre_name, t2 - t1)
                count = 0
                for row in cursor:
                    count += 1
                    morphs = igo_parse(row[desc_key])
                    morphs_filtered = [x[7] for x in morphs if x[1] in ["名詞", "動詞"]]

                    for morph in morphs_filtered:
                        if morph in genre_words[genre_name]:
                            genre_words[genre_name][morph] += 1
                        else:
                            genre_words[genre_name][morph] = 1

                t3 = time.time()
                print('{0}, {1}, {2} : {3}'.format(genre_id, genre_name, count, t3 - t2))



    finally:
        connection.close()

    print('length_n', len(genres))


    pickle.dump(genre_words, open('result/genres_2_b.out', 'wb'), -1)


    for key, value in sorted(genre_words.items(), key=lambda x:len(x[1]), reverse=True):
        print(key)
        for key1, value1 in sorted(value.items(), key=lambda x:x[1], reverse=True):
            print("-\t{0}\t{1}".format(key1, value1))
