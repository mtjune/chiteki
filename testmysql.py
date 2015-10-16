import pymysql
import yaml

setting = None
with open('mysql_setting.yml', 'r') as f:
    setting = yaml.load(f)


connection = pymysql.connect(host=setting['host'],
                             user=setting['user'],
                             password=setting['password'],
                             db='rakuten_recipe',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        sql = "select title from recipes limit 50;"
        cursor.execute(sql)
        for row in cursor:
            print(row['title'])
finally:
    connection.close()
