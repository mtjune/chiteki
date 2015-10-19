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
    cooktimes = [0, 0, 0, 0, 0, 0]
    cooktimes_count = [0, 0, 0, 0, 0, 0]
    try:
        with connection.cursor() as cursor:
            sql = "select cooktime_id, money_id from recipes;"
            cursor.execute(sql)
            for row in cursor:
                count += 1
                money = 0
                if row['money_id'] == 1:
                    money = 100
                elif row['money_id'] == 2:
                    money = 300
                elif row['money_id'] == 3:
                    money = 500
                elif row['money_id'] == 4:
                    money = 1000
                elif row['money_id'] == 5:
                    money = 2000
                elif row['money_id'] == 6:
                    money = 3000
                elif row['money_id'] == 7:
                    money = 5000
                elif row['money_id'] == 8:
                    money = 10000

                cooktimes[row['cooktime_id'] - 1] += money
                if money != 0:
                    cooktimes_count += 1

    finally:
        connection.close()


    print('count:', cooktimes_count)

    for i, v in enumerate(cooktimes):
        print('cooktime_id = {0}: {1}'.format(i + 1, v / cooktimes_count[i]))
