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
    n_v = {}
    try:
        with connection.cursor() as cursor:
            sql = "select ingredients.name as i_name, steps.memo as s_memo from ingredients join steps where ingredients.recipe_id = steps.recipe_id;"
            cursor.execute(sql)
            for row in cursor:
                count += 1
                i_name = row['i_name']
                morphs_memo = igo_parse(row['s_memo'])
                if i_name in [morph[0] for morph in morphs_memo if morph[1] == "名詞"]:
                    verbs = [morph[8] for morph in morphs_memo if morphs[1] == "動詞"]
                    if not n_v[i_name]:
                        n_v[i_name] = {}
                    for verb in verbs:
                        if n_v[i_name][verb]:
                            n_v[i_name][verb] += 1
                        else:
                            n_v[i_name][verb] = 1


    finally:
        connection.close()


    print('count:', count)
    print('length_n', len(n_v))



    i = 0
    for k, v in sorted(n_v.items(), key=lambda x:len(x[1]), reverse=True):
        print(k)
        j = 0
        for kk, vv in sorted(v.items(), key=lambda x:x[1], reverse=True):
            print("\t{0}:{1}".format(j, vv))
            j += 1
            if j > 10:
                break

        i += 1
        if i > 10:
            break
