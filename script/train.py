# coding=utf-8
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def prepare_data(input, target, maxlen=100):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    lengths_sess = [len(s[5]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    seqs_mid_sess = [inp[5] for inp in input]
    seqs_cat_sess = [inp[6] for inp in input]
    seqs_mid_tgt = [inp[7] for inp in input]
    seqs_cat_tgt = [inp[8] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_lengths_x = []
        new_seqs_mid_sess = []
        new_seqs_cat_sess = []
        new_seqs_mid_tgt = []
        new_seqs_cat_tgt = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_lengths_x.append(l_x)
        for l_sess, inp in zip(lengths_sess, input):
            if l_sess > 18:
                new_seqs_mid_sess.append(inp[5][l_sess-18:])
                new_seqs_cat_sess.append(inp[6][l_sess - 18:])
                new_seqs_mid_tgt.append(inp[7][l_sess - 18:])
                new_seqs_cat_tgt.append(inp[8][l_sess - 18:])
            else:
                new_seqs_mid_sess.append(inp[5])
                new_seqs_cat_sess.append(inp[6])
                new_seqs_mid_tgt.append(inp[7])
                new_seqs_cat_tgt.append(inp[8])
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        seqs_mid_sess = new_seqs_mid_sess
        seqs_cat_sess = new_seqs_cat_sess
        seqs_mid_tgt = new_seqs_mid_tgt
        seqs_cat_tgt = new_seqs_cat_tgt
        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    maxlen_sess = 18
    # neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    mid_sess_his = numpy.zeros((n_samples, maxlen_sess, 10)).astype('int64')
    cat_sess_his = numpy.zeros((n_samples, maxlen_sess, 10)).astype('int64')
    mid_sess_tgt = numpy.zeros((n_samples, 18))
    cat_sess_tgt = numpy.zeros((n_samples, 18))

    fin_mid_sess = numpy.array([inp[9] for inp in input])
    fin_cat_sess = numpy.array([inp[10] for inp in input])
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    sess_mask = numpy.zeros((n_samples, maxlen_sess)).astype('float32')
    for idx, [s_x, s_y, sess_x, sess_y, sess_x_tgt, sess_y_tgt] in enumerate(
            zip(seqs_mid, seqs_cat, seqs_mid_sess, seqs_cat_sess, seqs_mid_tgt, seqs_cat_tgt)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        sess_mask[idx, :lengths_sess[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        mid_sess_his[idx, :lengths_sess[idx]] = sess_x
        cat_sess_his[idx, :lengths_sess[idx]] = sess_y
        mid_sess_tgt[idx, :lengths_sess[idx]] = sess_x_tgt
        cat_sess_tgt[idx, :lengths_sess[idx]] = sess_y_tgt

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    # uid [1024*1]
    # mid [1024*1] 候选
    # mid_his [1024, 100] 历史商品序列
    # cat_his [1024, 100] 历史类别序列
    # mid_sess_his [1024, 18, 10] 10是session长度，18是session组成的序列长度
    # cat_sess_his
    # fin_mid_sess [1024, 10]
    # fin_cat_sess [1024, 10]

    return uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess, fin_cat_sess, numpy.array(
        target), numpy.array(lengths_x)


def eval(sess, test_data, model):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess, fin_cat_sess, target, sl = prepare_data(
            src, tgt)
        prob, loss, acc, aux_loss = model.calculate(sess,
                                                    [uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his,
                                                     cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess,
                                                     fin_cat_sess, target, sl])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        # model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum


def train(
        train_file="local_train",
        test_file="local_test",
        uid_voc="uid_voc.pkl",
        mid_voc="mid_voc.pkl",
        cat_voc="cat_voc.pkl",
        batch_size=2,
        maxlen=100,
        test_iter=1,
        save_iter=1000 * 5,
        model_type='DNN',
        seed=2,
):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        label_type = 1
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False,
                                  label_type=label_type)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, label_type=label_type)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DDPM':
            model = Model_DDPM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print("Invalid model_type : %s" % model_type)
            return
        print("Model: ", model_type)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()

        count()
        start_time = time.time()
        iter = 0
        lr = 0.001
        for itr in range(100):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                # print('batch_target = %d, itr=%d' % (len(tgt), itr))
                # time1 = time.time()
                uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess, fin_cat_sess, target, sl = prepare_data(
                    src, tgt, maxlen)
                # time2 = time.time()
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his,
                                                         cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
                                                         fin_mid_sess, fin_cat_sess, target, sl, lr])
                # time3 = time.time()
                # print('loss_time')
                # print(time3-time2)
                # print('data_process_time')
                # print(time2-time1)

                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
                if (iter % 50) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % (
                        iter, loss_sum / 100, accuracy_sum / 100, aux_loss_sum / 100))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % test_iter) == 0:
                    auc_, loss_, acc_, aux_ = eval(sess, test_data, model)
                    print(
                            'iter: %d --- test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- '
                            'test_aux_loss: %.4f' % (
                                iter, auc_, loss_, acc_, aux_))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % save_iter) == 0:
                    print('save model iter: %d' % (iter))
                    model.save(sess, model_path + "--" + str(iter))

                # lr *= 0.5


def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Prameter: ", total_parameters)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        SEED = int(sys.argv[3])
    else:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if sys.argv[1] == 'train':
        train(model_type=sys.argv[2], seed=SEED)
    else:
        print('do nothing...')
