

import argparse
import datetime
import multiprocessing
import random
import sys
import threading
import time
import os
import yaml
import csv
import shutil
import glob
import math

import numpy as np
import six
import six.moves.cPickle as pickle
from six.moves import queue

# from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers

from readmat import readDepthMap

parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('--arch', '-a', default='default', help='Convnet architecture(mini)')
parser.add_argument('--BATCHSIZE', '-B', type=int, default=25, help='Learning minibatch size')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--optimizer', '-o', default='adam', help='optimizer select')
parser.add_argument('--learningrate', '-l', default=0.01, type=float, help='momentumSGD Learning Rate')
parser.add_argument('--momentum', '-m', default=0.9, type=float, help='momentumSGD momentum')
parser.add_argument('--data', '-d', default='data', help='dataset directory')
parser.add_argument('--mean', default=None, help='mean image')
parser.add_argument('--ycbcr', '-y', action='store_true', default=False)
args = parser.parse_args()


# Prepare dataset

if os.path.isfile('config.yml'):
    CONFIG_FILE = 'config.yml'
else:
    CONFIG_FILE = 'config_default.yml'

config = None
with open(CONFIG_FILE, 'r') as f:
    config = yaml.load(f)

N_EPOCH = config['N_EPOCH']
BATCHSIZE = config['BATCHSIZE']
VAL_BATCHSIZE = config['VAL_BATCHSIZE']
WIDTH_IMG = config['WIDTH_IMG']
HEIGHT_IMG = config['HEIGHT_IMG']

print("N_EPOCH", N_EPOCH)
print("BATCHSIZE", BATCHSIZE)
print("WIDTH_IMG", WIDTH_IMG)
print("HEIGHT_IMG", HEIGHT_IMG)


NORM_MAX = 81

run_at =  datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')


config_write = {'program':__file__, 'N_EPOCH':N_EPOCH, 'BATCHSIZE':BATCHSIZE, 'GPU':args.gpu, 'OPTIMIZER':args.optimizer, 'RUN_at':run_at, 'learningrate':args.learningrate, 'momentum':args.momentum}
with open(LOGPATH + 'config.yml', 'w') as f:
    f.write(yaml.dump(config_write))



# Prepare model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, len(vocab)))

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



if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

xp = cuda.cupy if args.gpu >= 0 else np


def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units), dtype=np.float32), volatile=not train) for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
if args.optimizer == 'adam':
    optimizer = optimizers.Adam()
elif args.optimizer == 'adadelta':
    optimizer = optimizers.AdaDelta()
elif args.optimizer == 'adagrad':
    optimizer = optimizers.AdaGrad()
elif args.optimizer == 'rmsprop':
    optimizer = optimizers.RMSprop()
elif args.optimizer == 'msgd':
    optimizer = optimizers.MomentumSGD(lr=args.learningrate, momentum=args.momentum)
else:
    optimizer = optimizers.Adam()


optimizer.setup(model)

train_list = []
with file(path_train + 'list.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        train_list.append(row)

val_list = []
with file(path_val + 'list.csv') as f:
    readcsv = csv.reader(f)
    for row in readcsv:
        val_list.append(row)


train_num = len(train_list) - (len(train_list) % BATCHSIZE)
val_num = len(val_list) - (len(val_list) % VAL_BATCHSIZE)


def add_record(row_array, record_type):
    if record_type == 'time':
        filename = LOGPATH + 'record_time.csv'
    else:
        filename = LOGPATH + 'record_loss.csv'

    with open(filename, 'a') as f:
        record = csv.writer(f, lineterminator='\n')
        record.writerow(row_array)


# ------------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and trainer.
# These communicate with each other via Queue.
data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

if args.mean:
    with open(args.mean, 'rb') as f:
        image_mean = pickle.load(f)


def read_image(x_name, y_name):
    if args.ycbcr:
        image_x = np.asarray(Image.open(x_name).convert("YCbCr").resize((WIDTH_IMG, HEIGHT_IMG)), dtype=np.float32).transpose(2, 0, 1)
    else:
        image_x = np.asarray(Image.open(x_name).resize((WIDTH_IMG, HEIGHT_IMG)), dtype=np.float32).transpose(2, 0, 1)

    if CONV_FLAG:
        image_y = np.asarray(readDepthMap(y_name).resize((WIDTH_IMG, HEIGHT_IMG)), dtype=np.float32).reshape((HEIGHT_IMG, WIDTH_IMG, 1)).transpose(2, 0, 1)
        image_y1 = np.ndarray((1, HEIGHT_Y1, WIDTH_Y1), dtype = np.float32)
        for h in range(HEIGHT_Y1):
            for w in range(WIDTH_Y1):
                image_y1[0, h, w] = np.average(image_y[0, h:h + STRIDE, w:w + STRIDE])
    else:
        image_y = np.asarray(readDepthMap(y_name).resize((WIDTH_IMG, HEIGHT_IMG)), dtype=np.float32).reshape(-1)
        image_y1 = None
    if args.mean:
        output_x = image_x - image_mean

    output_x = image_x / 255
    output_y = image_y / NORM_MAX
    output_y1 = image_y1 / NORM_MAX
    return (output_x, output_y, output_y1)

# Data feeder


def feed_data():
    i = 0
    count = 0

    x_batch = np.ndarray((BATCHSIZE, 3, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
    val_x_batch = np.ndarray((VAL_BATCHSIZE, 3, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)

    if CONV_FLAG:
        y_batch = np.ndarray((BATCHSIZE, 1, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
        y1_batch = np.ndarray((BATCHSIZE, 1, HEIGHT_Y1, WIDTH_Y1), dtype=np.float32)
        val_y_batch = np.ndarray((VAL_BATCHSIZE, 1, HEIGHT_IMG, WIDTH_IMG), dtype=np.float32)
        val_y1_batch = np.ndarray((VAL_BATCHSIZE, 1, HEIGHT_Y1, WIDTH_Y1), dtype=np.float32)
    else:
        y_batch = np.ndarray((BATCHSIZE, HEIGHT_IMG * WIDTH_IMG), dtype=np.float32)
        val_y_batch = np.ndarray((VAL_BATCHSIZE, HEIGHT_IMG * WIDTH_IMG), dtype=np.float32)

    batch_pool = [None] * BATCHSIZE
    val_batch_pool = [None] * VAL_BATCHSIZE
    pool = multiprocessing.Pool()
    # data_q.put('train')
    for epoch in six.moves.range(1, 1 + N_EPOCH):
        # print('epoch', epoch, file=sys.stderr)
        # print('learning rate', optimizer.lr, file=sys.stderr)
        data_q.put('train')
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            x_name, y_name = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, (x_name, y_name))
            i += 1

            if i == BATCHSIZE:
                for j, x in enumerate(batch_pool):
                    (x_batch[j], y_batch[j], y1_batch[j]) = x.get()
                data_q.put((x_batch.copy(), y_batch.copy(), y1_batch.copy()))
                i = 0

            count += 1

        data_q.put('val')
        j = 0
        for x_name, y_name in val_list:
            val_batch_pool[j] = pool.apply_async(read_image, (x_name, y_name))
            j += 1

            if j == VAL_BATCHSIZE:
                for k, x in enumerate(val_batch_pool):
                    (val_x_batch[k], val_y_batch[k], val_y1_batch[k]) = x.get()
                data_q.put((val_x_batch.copy(), val_y_batch.copy(), val_y1_batch.copy()))
                j = 0
        # data_q.put('train')

        if args.optimizer == 'msgd':
            optimizer.lr *= 0.97

    pool.close()
    pool.join()
    data_q.put('end')

# Logger

def show_progress(value, max_value):
    MAX = 20
    sharp_num = value * MAX / max_value
    progress = '[' + ('#' * sharp_num) + ('-' * (20 - sharp_num)) + ']'
    progress += '({}/{})'.format(value, max_value)
    return progress

def log_result():
    train_count = 0
    train_cur_loss = 0
    epoch = 0


    t_train_start = t_val_start = None
    t_train = []
    t_val = []
    while True:
        result = res_q.get()
        if result == 'end':
            t_val.append(time.time() - t_val_start)
            add_record([epoch, t_train[epoch - 1], t_val[epoch - 1]], 'time')
            mean_loss = val_loss * BATCHSIZE / val_count
            mag_loss = math.log(math.sqrt(mean_loss) * NORM_MAX, 10)
            add_record([epoch, mag_loss], 'loss')
            print()
            print('val mag loss :{}'.format(mag_loss))
            print()
            break
        elif result == 'train':
            if epoch > 0:
                t_val.append(time.time() - t_val_start)
                add_record([epoch, t_train[epoch - 1], t_val[epoch - 1]], 'time')
                mean_loss = val_loss * BATCHSIZE / val_count
                mag_loss = math.log(math.sqrt(mean_loss) * NORM_MAX, 10)
                add_record([epoch, mag_loss], 'loss')
                print()
                print('val mag loss :{}'.format(mag_loss))
                print()

            epoch += 1
            train = True
            print('epoch:' + str(epoch))
            train_count = 0
            t_train_start = time.time()
            continue
        elif result == 'val':
            t_train.append(time.time() - t_train_start)

            train = False
            print()
            val_count = val_loss = 0
            t_val_start = time.time()
            continue

        loss = result
        if train:
            train_count += BATCHSIZE

            progress = '\rtrain\t' + show_progress(train_count, train_num)
            sys.stdout.write(progress)
            sys.stdout.flush()

            train_cur_loss += loss

        else:
            val_count += VAL_BATCHSIZE

            progress = '\rval\t' + show_progress(val_count, val_num)
            sys.stdout.write(progress)
            sys.stdout.flush()

            val_loss += loss

# Trainer


def train_loop():
    graph_generated = False
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'train':  # restart training
            res_q.put('train')
            train = True
            continue
        elif inp == 'val':  # start validation
            pickle.dump(model, open(LOGPATH + 'model', 'wb'), -1)
            res_q.put('val')
            train = False
            continue


        x = xp.asarray(inp[0])
        y = xp.asarray(inp[1])
        y1 = xp.asarray(inp[2])

        if train:
            optimizer.zero_grads()
            loss = model.forward(x, y, y1, train=True)
            loss.backward()
            optimizer.update()

        else:
            loss = model.forward(x, y, y1, train=False)

        res_q.put(float(cuda.to_cpu(loss.data)))
        del loss, x, y, y1

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
pickle.dump(model, open(LOGPATH + 'model', 'wb'), -1)

os.rename(LOGDIRNAME, LOGDIRNAME + '_success')

print(LOGDIRNAME)
