# encoding: utf-8
import pymysql
import yaml

import pandas as pd
import numpy as np

import igo

CHAR_CODE = "utf-8"
DIC_DIR = "~/workspace/tmp/igo_ipadic"

def mecabparse(text):
    mt = MeCab.Tagger()
    # text = text.encode(CHAR_CODE)

    res = mt.parse(text)
    outputs = []

    lines = res.split("\n")

    for line in lines[0:-2]	:
        word, feature = line.split("\t", 2)
        outputs.append([word] + feature.split(","))

    return outputs



if __name__ == '__main__':

    tagger = igo.Tagger(DIC_DIR)

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
            sql = "select title from recipes limit 50;"
            cursor.execute(sql)
            for row in cursor:
                words = mecabparse(row['title'])
                words_n = [w[0] for w in words if w[1] is "名詞"]
                for word in words_n:
                    if word in nns:
                        nns[word] += 1
                    else:
                        nns[word] = 1

    finally:
        connection.close()

    print(nns)
