# encoding: utf-8


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
parser.add_argument('--vocab', '-c', default='result/vocab_dic_1_b.out')
parser.add_argument('--recipe', '-r', default='result/recipe_ids_1_b.out')
parser.add_argument('--model', '-m', default='result/lstm.model.out')
parser.add_argument('--output', '-o', default='result/lstm')
parser.add_argument('--gpu', '-g', default=-1, type=int)
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np


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

setting = None
with open('mysql_setting.yml', 'r') as f:
    setting = yaml.load(f)
connection = pymysql.connect(host=setting['host'],
                             user=setting['user'],
                             password=setting['password'],
                             db='rakuten_recipe',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.SSCursor)



n_epoch = 39   # number of epochs
n_units = 650  # number of units per layer
batchsize = 10   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip

# Prepare dataset
vocab = None
with open(args.vocab, 'rb') as f:
    vocab = pickle.load(f)

recipe_ids = None
with open(args.recipe, 'rb') as f:
    recipe_ids = pickle.load(f)


def load_data(text):
    words = igo_parse(text)
    dataset = []
    for i, word in enumerate(words):
        if word not in vocab:
            print('{} not in vocab'.format(word))
            continue

        dataset.append(vocab[word])
    output = np.asarray(dataset, dtype=np.int32)
    return output


train_text = ''
valid_text = ''
test_text = ''

try:
    for recipe_id in recipe_ids[:100]:
        with connection.cursor() as cursor:
            sql = "select position, memo from steps where recipe_id = {}".format(recipe_id)
            cursor.execute(sql)
            result = cursor.fetchall()

            for _, text in sorted(result, key=lambda x:x[0]):
                train_text = train_text + text

    for recipe_id in recipe_ids[100:150]:
        with connection.cursor() as cursor:
            sql = "select position, memo from steps where recipe_id = {}".format(recipe_id)
            cursor.execute(sql)
            result = cursor.fetchall()

            for _, text in sorted(result, key=lambda x:x[0]):
                valid_text = valid_text + text

    for recipe_id in recipe_ids[150:200]:
        with connection.cursor() as cursor:
            sql = "select position, memo from steps where recipe_id = {}".format(recipe_id)
            cursor.execute(sql)
            result = cursor.fetchall()

            for _, text in sorted(result, key=lambda x:x[0]):
                test_text = test_text + text

finally:
    connection.close()

train_data = load_data(train_text)
valid_data = load_data(valid_text)
test_data = load_data(test_text)
print('#vocab =', len(vocab))

# Prepare RNNLM model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, len(vocab)))
for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward_one_step(x_data, y_data, state, train=True):
    # Neural net architecture
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(F.dropout(h2, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, F.softmax_cross_entropy(y, t)


def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units),
                                            dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model)


# Evaluation routine


def evaluate(dataset):
    sum_log_perp = xp.zeros(())
    state = make_initial_state(batchsize=1, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = xp.asarray(dataset[i:i + 1])
        y_batch = xp.asarray(dataset[i + 1:i + 2])
        state, loss = forward_one_step(x_batch, y_batch, state, train=False)
        sum_log_perp += loss.data.reshape(())

    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))


# Learning loop
whole_len = train_data.shape[0]
jump = whole_len // batchsize
cur_log_perp = xp.zeros(())
epoch = 0
start_at = time.time()
cur_at = start_at
state = make_initial_state()
accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
print('going to train {} iterations'.format(jump * n_epoch))
for i in six.moves.range(jump * n_epoch):
    x_batch = xp.array([train_data[(jump * j + i) % whole_len]
                        for j in six.moves.range(batchsize)])
    y_batch = xp.array([train_data[(jump * j + i + 1) % whole_len]
                        for j in six.moves.range(batchsize)])
    state, loss_i = forward_one_step(x_batch, y_batch, state)
    accum_loss += loss_i
    cur_log_perp += loss_i.data.reshape(())

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))

        optimizer.clip_grads(grad_clip)
        optimizer.update()

    if (i + 1) % 1000 == 0:
        now = time.time()
        throuput = 1000. / (now - cur_at)
        perp = math.exp(cuda.to_cpu(cur_log_perp) / 1000)
        print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
            i + 1, perp, throuput))
        add_record([i + 1, perp], 'loss_train')
        add_record([i + 1, now - start_at, throuput], 'time')
        cur_at = now
        cur_log_perp.fill(0)

    if (i + 1) % jump == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        perp = evaluate(valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        add_record([epoch, i + 1, perp], 'loss_valid')
        cur_at += time.time() - now  # skip time of evaluation

        if epoch >= 6:
            optimizer.lr /= 1.2
            print('learning rate =', optimizer.lr)

        with open(args.model, 'wb') as f:
            pickle.dump(model, f, -1)
        print('saved')

    sys.stdout.flush()

with open(args.model, 'wb') as f:
    pickle.dump(model, f, -1)
print('saved')

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)
