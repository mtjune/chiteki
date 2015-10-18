# encoding: utf-8
import pymysql
import yaml

import pandas as pd
import numpy as np

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
                                 db='rakuten_recipe',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)


    nns = {}

    try:
        with connection.cursor() as cursor:
            sql = "select title from recipes;"
            cursor.execute(sql)
            for row in cursor:
                words = igo_parse(row['title'])
                words_n = [w[0] for w in words if w[1] == "名詞"]
                for word in words_n:
                    if word in nns:
                        nns[word] += 1
                    else:
                        nns[word] = 1

    finally:
        connection.close()


    nns_limited = nns[0:50]

    for k, v in sorted(nns_limited.items(), key=lambda x:x[1]):
        print(k, v)
