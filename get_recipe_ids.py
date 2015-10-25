# encoding: utf-8

import pymysql
import yaml

import pandas as pd
import numpy as np
import six.moves.cPickle as pickle

if __name__ == '__main__':

    setting = None
    with open('mysql_setting.yml', 'r') as f:
        setting = yaml.load(f)


    connection = pymysql.connect(host=setting['host'],
                                 user=setting['user'],
                                 password=setting['password'],
                                 db='rakuten_recipe',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.SSCursor)


    recipe_ids = None


    try:
        with connection.cursor() as cursor:

            sql = "select recipe_id from recipes where small_category = 'チーズケーキ';"
            print("query :", sql)
            cursor.execute(sql)
            print("query complete")

            recipe_ids = [row[0] for row in cursor]
            recipe_num = len(recipe_ids)


    finally:
        connection.close()

    print('process end', len(recipe_ids))

    with open('result/recipe_ids_1_b.out', 'wb') as f:
        pickle.dump(recipe_ids, f, -1)


    print("complete!")
