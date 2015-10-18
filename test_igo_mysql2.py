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


    count = 0
    categories = {}
    try:
        with connection.cursor() as cursor:
            sql = "select large_category from recipes;"
            cursor.execute(sql)
            for row in cursor:
                count += 1
                word = row['large_category']
                if word in categories:
                    categories[word] += 1
                else:
                    categories[word] = 1

    finally:
        connection.close()


    print('count:', count)
    print('dictlen', len(categories))

    i = 0
    for k, v in sorted(categories.items(), key=lambda x:x[1], reverse=True):
        print(k, v)
        i += 1
        if i > 50:
            break
