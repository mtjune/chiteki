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
                             cursorclass=pymysql.cursors.SSCursor)

try:
    with connection.cursor() as cursor:
        sql = "select recipe_id, title from recipes limit 50;"
        cursor.execute(sql)
        for desc in cursor.description:
            print(desc)
finally:
    connection.close()
