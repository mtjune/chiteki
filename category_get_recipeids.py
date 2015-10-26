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


    category_recipe_ids = None

    categories = ['おかず・おつまみ', 'おやつ', 'お弁当', 'ソース・調味料・ジャム', '主食', '国・地域別料理', '季節・イベント', '目的別', '飲み物']
    category_recipe_ids = {x:[] for x in categories}

    recipe_num = {}

    try:
        for category in categories:
            with connection.cursor() as cursor:

                sql = "select recipe_id from recipes where large_category = '{}' order by rand() limit 1000;".format(category)
                print("query :", sql)
                cursor.execute(sql)
                print("query complete")

                category_recipe_ids[category] = [row[0] for row in cursor]


    finally:
        connection.close()

    print('process end')

    for category, recipe_ids in category_recipe_ids.items():
        print(category, len(recipe_ids))


    with open('result/category_recipe_ids_b.out', 'wb') as f:
        pickle.dump(category_recipe_ids, f, -1)


    print("complete!")
