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
                                 cursorclass=pymysql.cursors.SSCursor)


    word_scores = {}
    try:
        with connection.cursor() as cursor:

            sql = "select point, description from review;"
            print("query :", sql)
            cursor.execute(sql)
            print("query complete")
            cur_length = cursor.rowcount
            count = 0
            for row in cursor:
                if not row[0] in ['0', '1', '2', '3', '4', '5']:
                    continue
                score = float(row[0])
                score = (score / 2.5) - 1.0

                morphs = igo_parse(row[1])
                morphs_filtered = [x[7] for x in morphs if x[1] in ["名詞", "形容詞", "副詞", "動詞"]]

                for morph in morphs_filtered:
                    if morph in word_scores:
                        a, b = word_scores[morph]
                        word_scores[morph] = (a + score, b + 1)
                    else:
                        word_scores[morph] = (score, 1)

                if count % 10000 == 0:
                    print("end: {0}/{1}".format(count, cur_length))
                if count % 100000 == 0:
                    pickle.dump(word_scores, open('result/word_scores_2_b.out', 'wb'), -1)
                    print("saved : {}".format(count))
                count += 1


    finally:
        connection.close()



    pickle.dump(word_scores, open('result/word_scores_2_b.out', 'wb'), -1)

    print("complete!")
