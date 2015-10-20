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
    tags = {}
    try:
        with connection.cursor() as cursor:
            sql = "select tag1, tag2, tag3, tag4 from recipes;"
            cursor.execute(sql)
            for row in cursor:
                count += 1
                if 'tag1' in row:
                    if row['tag1'] in tags:
                        tags[row['tag1']] += 1
                    else:
                        tags[row['tag1']] = 1
                if 'tag2' in row:
                    if row['tag2'] in tags:
                        tags[row['tag2']] += 1
                    else:
                        tags[row['tag2']] = 1
                if 'tag3' in row:
                    if row['tag3'] in tags:
                        tags[row['tag3']] += 1
                    else:
                        tags[row['tag3']] = 1
                if 'tag4' in row:
                    if row['tag4'] in tags:
                        tags[row['tag4']] += 1
                    else:
                        tags[row['tag4']] = 1


    finally:
        connection.close()


    print('count:', count)
    print('length_tags', len(tags))


    for k, v in sorted(n_v.items(), key=lambda x:x[1], reverse=True)[0:50]:
        print(k, v)
