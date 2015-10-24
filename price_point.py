# encoding: utf-8

# 市場データから，レビューよりどの要素がその商品の評価に関わるか

import pymysql
import yaml
import time

import pandas as pd
import numpy as np
import six.moves.cPickle as pickle


import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


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




    prices = []
    points = []

    t0 = None
    t1 = None
    t2 = None
    try:
        with connection.cursor() as cursor:

            t0 = time.time()
            sql = "select item_price, point from review where purchased = '1' limit 1000000;"
            cursor.execute(sql)

            t1 = time.time()

            rowlength = cursor.rowcount
            count = 0
            for row in cursor:
                prices.append(int(row[0]))
                points.append(int(row[1]))

                if count % 100000 == 0:
                    print("{0}/{1}".format(count, rowlength))
                count += 1


            t2 = time.time()

    finally:
        connection.close()

    plt.plot(prices, points, 'o')
    plt.savefig("result/fig1.png")


    # pickle.dump(genres, open('result/genres_only_b.out', 'wb'), -1)
    # print('saved')

    print('1 - 0 : {}'.format(t1 - t0))
    print('2 - 1 : {}'.format(t2 - t1))
