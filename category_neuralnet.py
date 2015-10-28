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
parser.add_argument('--recipe', '-r', default='result/category_recipe_ids_b.out')
parser.add_argument('--model', '-m', default='result/category.model.out')
parser.add_argument('--output', '-o', default='result/category')
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
    elif record_type == 'loss_valid':
        filename = args.output + '_loss_valid.csv'
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

category_recipe_ids = None
with open(args.recipe, 'rb') as f:
    category_recipe_ids = pickle.load(f)


count_novocab = 0
def load_data(recipe_id):
    global count_novocab
    output_data = np.zeros((len(vocab),), dtype=np.int32)
    flag_novocab = True
    with connection.cursor() as cursor:
        sql = "select title from recipes where recipe_id = {};".format(recipe_id)
        cursor.execute(sql)
        title = cursor.fetchone()[0]

        words = igo_parse(title)
        for word in words:
            if word in vocab:
                output_data[vocab[word]] = 1
                flag_novocab = False
    if flag_novocab:
        count_novocab += 1
    return output_data




train_data = []
valid_data = []
for category, recipe_ids in category_recipe_ids.items():
    for recipe_id in recipe_ids[0:900]:
        train_data.append((recipe_id, categories[category]))

    for recipe_id in recipe_ids[900:1000]:
        valid_data.append((recipe_id, categories[category]))


n_train = len(train_data)
n_valid = len(valid_data)

n_epoch = 40   # number of epochs
n_units = 500  # number of units per layer
batchsize = 25   # minibatch size
batchsize_valid = 50



# Prepare RNNLM model
model = chainer.FunctionSet(l1=F.Linear(len(vocab), n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, len(categories)),
                            lae=F.Linear(n_units, len(vocab)))


def forward(x_data, y_data, train=True):
    # Neural net architecture
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)

    h = F.dropout(F.relu(model.l1(x)), ratio=0.1, train=train)
    h = F.dropout(F.relu(model.l2(h)), train=train)
    y = model.l3(h)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def forward_ae(x_data, train=True):
    x = chainer.Variable(x_data, volatile=not train)
    t = x

    h = F.relu(model.l1(x))
    y = F.sigmoid(model.lae(h))

    return F.mean_squared_error(y, t)


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)


# Pre train
pretrain_epoch = 10
for epoch in six.moves.range(1, pretrain_epoch + 1):
    print('pretrain_epoch', epoch)
    # training
    perm = np.random.permutation(n_train)
    sum_loss = 0

    for i in six.moves.range(0, n_train, batchsize):
        x_batch = np.zeros((batchsize, len(vocab)), dtype=np.float32)

        for j in six.moves.range(batchsize):
            recipe_id, label = train_data[perm[i + j]]
            x_batch[j, :] = load_data(recipe_id)

        optimizer.zero_grads()
        loss = forward_ae(x_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * batchsize


    print('train mean loss={}'.format(sum_loss / n_train))


for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, n_valid, batchsize_valid):
            x_batch = np.zeros((batchsize_valid, len(vocab)), dtype=np.float32)
            y_batch = np.zeros((batchsize_valid,), dtype=np.int32)

            for j in six.moves.range(batchsize_valid):
                recipe_id, label = valid_data[i + j]
                x_batch[j, :] = load_data(recipe_id)
                y_batch[j] = label

            loss, acc = forward(x_batch, y_batch, train=False)

            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)

        valid_at = time.time()
        print('valid mean loss={}, accuracy={}'.format(sum_loss / n_valid, sum_accuracy / n_valid))


    start_at = time.time()
    # training
    perm = np.random.permutation(n_train)
    sum_accuracy = 0
    sum_loss = 0

    for i in six.moves.range(0, n_train, batchsize):
        x_batch = np.zeros((batchsize, len(vocab)), dtype=np.float32)
        y_batch = np.zeros((batchsize,), dtype=np.int32)

        for j in six.moves.range(batchsize):
            recipe_id, label = train_data[perm[i + j]]
            x_batch[j, :] = load_data(recipe_id)
            y_batch[j] = label

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    train_at = time.time()
    print('train mean loss={}, accuracy={}'.format(sum_loss / n_train, sum_accuracy / n_train))
    add_record([epoch, sum_loss / n_train, sum_accuracy / n_train], 'loss_train')
    train_novocab = count_novocab
    count_novocab = 0

    # valid
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, n_valid, batchsize_valid):
        x_batch = np.zeros((batchsize_valid, len(vocab)), dtype=np.float32)
        y_batch = np.zeros((batchsize_valid,), dtype=np.int32)

        for j in six.moves.range(batchsize_valid):
            recipe_id, label = valid_data[i + j]
            x_batch[j, :] = load_data(recipe_id)
            y_batch[j] = label

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    valid_at = time.time()
    print('valid mean loss={}, accuracy={}'.format(sum_loss / n_valid, sum_accuracy / n_valid))
    add_record([epoch, sum_loss / n_valid, sum_accuracy / n_valid], 'loss_valid')
    add_record([epoch, train_at - start_at, valid_at - train_at], 'time')

    valid_novocab = count_novocab
    count_novocab = 0
    print('train novocab : {} , valid_novocab : {}'.format(train_novocab, valid_novocab))

    with open(args.model, 'wb') as f:
        pickle.dump(model, f, -1)

    print('saved', args.model)

print('all end')
