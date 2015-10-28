import argparse
import math
import sys
import time
import yaml
import csv

import pymysql

import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import igo


parser = argparse.ArgumentParser()
parser.add_argument('--vocab', '-c', default='result/category_vocab_b.out')
parser.add_argument('--model', '-m', default='result/category.model.out')
parser.add_argument('--recipe', '-r', default='result/category_recipe_ids_b.out')
parser.add_argument('--output', '-o', default='result/category_test')
parser.add_argument('--gpu', '-g', default=-1, type=int)
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np


setting = None
with open('mysql_setting.yml', 'r') as f:
    setting = yaml.load(f)
connection = pymysql.connect(host=setting['host'],
                             user=setting['user'],
                             password=setting['password'],
                             db='rakuten_recipe',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.SSCursor)


def add_record(row_array, record_type):
    if record_type == 'time':
        filename = args.output + '_time.csv'
    elif record_type == 'loss_test':
        filename = args.output + '_loss_test.csv'
    elif record_type == 'loss_train':
        filename = args.output + '_loss_train.csv'

    with open(filename, 'a') as f:
        record = csv.writer(f, lineterminator='\n')
        record.writerow(row_array)

DIC_DIR = "/home/yamajun/workspace/tmp/igo_ipadic"
tagger = igo.tagger.Tagger(DIC_DIR)
def igo_parse(text):
    words = tagger.parse(text)
    outputs = [word.surface for word in words]
    return outputs



# Prepare dataset

categories = {'おかず・おつまみ':0, 'おやつ':1, 'お弁当':2, 'ソース・調味料・ジャム':3, '主食':4, '国・地域別料理':5, '季節・イベント':6, '目的別':7, '飲み物':8}

vocab = None
with open(args.vocab, 'rb') as f:
    vocab = pickle.load(f)


count_invalid = 0
count_novocab = 0
def load_data(recipe_id):
    global count_invalid, count_novocab
    output_data = np.zeros((len(vocab),), dtype=np.int32)
    with connection.cursor() as cursor:
        sql = "select title from recipes where recipe_id = {};".format(recipe_id)
        cursor.execute(sql)
        title = cursor.fetchone()[0]

        words = igo_parse(title)
        flag_invalid = True
        flag_novocab = True
        for word in words:
            flag_invalid = False
            if word in vocab:
                flag_novocab = False
                output_data[vocab[word]] = 1

        if flag_invalid:
            count_invalid += 1
        if flag_novocab:
            count_novocab += 1

    return output_data



# test_data = []
# with connection.cursor() as cursor:
#     sql = "select recipe_id, large_category from recipes"
#     cursor.execute(sql)
#
#     for row in cursor:
#         test_data.append((row[0], categories[row[1]]))

category_recipe_ids = None
with open(args.recipe, 'rb') as f:
    category_recipe_ids = pickle.load(f)


test_data = []
for category, recipe_ids in category_recipe_ids.items():
    for recipe_id in recipe_ids[900:1000]:
        test_data.append((recipe_id, categories[category]))


n_test = len(test_data)
print('n_test:{}'.format(n_test))

n_epoch = 40   # number of epochs
n_units = 800  # number of units per layer
batchsize_test = 10



# Prepare RNNLM model
model = None
with open(args.model, 'rb') as f:
    model = pickle.load(f)

match_mat = np.zeros((len(categories), len(categories)), dtype=np.int32)

def forward(x_data, y_data, train=True):
    global match_mat
    # Neural net architecture
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)

    h = F.dropout(F.relu(model.l1(x)), train=train)
    h = F.dropout(F.relu(model.l2(h)), train=train)
    y = model.l3(h)

    y_softed = F.softmax(y).data
    for i in range(batchsize_test):
        y_ind = np.argmax(y_softed[i, :])
        t_ind = y_data[i]
        match_mat[t_ind, y_ind] += 1

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)


# test
sum_accuracy = 0
sum_loss = 0
n_test_clip = (n_test // batchsize_test) * batchsize_test
n_show = n_test_clip / batchsize_test // 20
for i in six.moves.range(0, n_test_clip, batchsize_test):
    x_batch = np.zeros((batchsize_test, len(vocab)), dtype=np.float32)
    y_batch = np.zeros((batchsize_test,), dtype=np.int32)

    for j in six.moves.range(batchsize_test):
        recipe_id, label = test_data[i + j]
        x_batch[j, :] = load_data(recipe_id)
        y_batch[j] = label

    loss, acc = forward(x_batch, y_batch, train=False)

    sum_loss += float(loss.data) * batchsize_test
    sum_accuracy += float(acc.data) * batchsize_test

    # if i % (n_show * batchsize_test ) == 0:
    #     print('test {} / {} mean loss={}, accuracy={}'.format(i, n_test_clip, sum_loss / (i + batchsize_test), sum_accuracy / (i + batchsize_test)))
    #     add_record([i, sum_loss / (i + batchsize_test), sum_accuracy / (i + batchsize_test)], 'loss_test')
    #
    #     with open('category_test_mat_2.out', 'wb') as f:
    #         pickle.dump(match_mat, f, -1)
    #
    #     print('invalid_recipe : {} , novocab_recipe : {}'.format(count_invalid, count_novocab))

test_at = time.time()
print('test end {} mean loss={}, accuracy={}'.format(n_test_clip, sum_loss / n_test_clip, sum_accuracy / n_test_clip))
add_record([n_test_clip, sum_loss / n_test_clip, sum_accuracy / n_test_clip], 'loss_test')
with open('category_test_mat.out', 'wb') as f:
    pickle.dump(match_mat, f, -1)
print('invalid_recipe : {} , novocab_recipe : {}'.format(count_invalid, count_novocab))

print('all end')
