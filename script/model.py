# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
from logger import set_logger
import numpy as np
import math

logger = set_logger()
#### CAN config #####
weight_emb_w = [[16, 8], [8, 4]]
weight_emb_b = [0, 0]
print(weight_emb_w, weight_emb_b)
orders = 3
order_indep = False  # True
WEIGHT_EMB_DIM = (sum([w[0] * w[1] for w in weight_emb_w]) + sum(weight_emb_b))  # * orders
INDEP_NUM = 1
if order_indep:
    INDEP_NUM *= orders

print("orders: ", orders)
CALC_MODE = "can"
device = '/gpu:0'


#### CAN config #####

def gen_coaction(ad, his_items, dim, mode="can", mask=None):
    weight, bias = [], []
    idx = 0
    weight_orders = []
    bias_orders = []
    for i in range(orders):
        for w, b in zip(weight_emb_w, weight_emb_b):
            weight.append(tf.reshape(ad[:, idx:idx + w[0] * w[1]], [-1, w[0], w[1]]))
            idx += w[0] * w[1]
            if b == 0:
                bias.append(None)
            else:
                bias.append(tf.reshape(ad[:, idx:idx + b], [-1, 1, b]))
                idx += b
        weight_orders.append(weight)
        bias_orders.append(bias)
        if not order_indep:
            break

    if mode == "can":
        out_seq = []
        hh = []
        for i in range(orders):
            hh.append(his_items ** (i + 1))
        # hh = [sum(hh)]
        for i, h in enumerate(hh):
            if order_indep:
                weight, bias = weight_orders[i], bias_orders[i]
            else:
                weight, bias = weight_orders[0], bias_orders[0]
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)
                if b is not None:
                    h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)
                out_seq.append(h)
        out_seq = tf.concat(out_seq, 2)
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
            out_seq = out_seq * mask
    out = tf.reduce_sum(out_seq, 1)
    # if keep_fake_carte_seq and mode=="emb":
    #    return out, out_seq
    return out, None


class Model(object):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_softmax=True, use_coaction=False, use_cartes=False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')

            self.mid_sess_his = tf.placeholder(tf.int32, [None, 18, 10], name='mid_sess_his')  # [1024, 18, 10]
            self.cat_sess_his = tf.placeholder(tf.int32, [None, 18, 10], name='cat_sess_his')
            self.sess_mask = tf.placeholder(tf.int32, [None, 18], name='sess_mask')
            self.mid_sess_tgt = tf.placeholder(tf.int32, [None, 18], name='mid_sess_tgt')  # [1024, 18]
            self.cat_sess_tgt = tf.placeholder(tf.int32, [None, 18], name='cat_sess_tgt')
            self.fin_mid_sess = tf.placeholder(tf.int32, [None, 10], name='fin_mid_sess')  # [1024, 10]
            self.fin_cat_sess = tf.placeholder(tf.int32, [None, 10], name='fin_cat_sess')

            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.cl_label = tf.placeholder(tf.float32, [None, None], name='cl_label_ph')
            # self.carte_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='carte_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling = use_negsampling
            self.use_softmax = False  # use_softmax
            self.use_coaction = use_coaction
            self.use_cartes = use_cartes
            print("args:")
            print("negsampling: ", self.use_negsampling)
            print("softmax: ", self.use_softmax)
            print("co-action: ", self.use_coaction)
            print("carte: ", self.use_cartes)
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                         name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cate_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cate_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)

            self.cate_embeddings_var = tf.get_variable("cate_embedding_var", [n_cate, EMBEDDING_DIM])
            tf.summary.histogram('cate_embeddings_var', self.cate_embeddings_var)
            self.cate_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cate_batch_ph)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cate_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var,
                                                                            self.noclk_cate_batch_ph)

            ###  co-action ###
            if self.use_coaction:
                ph_dict = {
                    "item": [self.mid_batch_ph, self.mid_his_batch_ph, self.mid_his_batch_embedded],
                    "cate": [self.cate_batch_ph, self.cate_his_batch_ph, self.cate_his_batch_embedded]
                }
                self.mlp_batch_embedded = []
                with tf.device(device):
                    self.item_mlp_embeddings_var = tf.get_variable("item_mlp_embedding_var",
                                                                   [n_mid, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)
                    self.cate_mlp_embeddings_var = tf.get_variable("cate_mlp_embedding_var",
                                                                   [n_cate, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)

                    self.mlp_batch_embedded.append(
                        tf.nn.embedding_lookup(self.item_mlp_embeddings_var, ph_dict['item'][0]))
                    self.mlp_batch_embedded.append(
                        tf.nn.embedding_lookup(self.cate_mlp_embeddings_var, ph_dict['cate'][0]))

                    self.input_batch_embedded = []
                    self.item_input_embeddings_var = tf.get_variable("item_input_embedding_var",
                                                                     [n_mid, weight_emb_w[0][0] * INDEP_NUM],
                                                                     trainable=True)
                    self.cate_input_embeddings_var = tf.get_variable("cate_input_embedding_var",
                                                                     [n_cate, weight_emb_w[0][0] * INDEP_NUM],
                                                                     trainable=True)
                    self.input_batch_embedded.append(
                        tf.nn.embedding_lookup(self.item_input_embeddings_var, ph_dict['item'][1]))
                    self.input_batch_embedded.append(
                        tf.nn.embedding_lookup(self.cate_input_embeddings_var, ph_dict['cate'][1]))

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cate_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cate_his_batch_embedded[:, :, 0, :]],
                -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                                 2 * EMBEDDING_DIM])  # cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cate_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

        self.cross = []
        if self.use_coaction:
            input_batch = self.input_batch_embedded
            tmp_sum, tmp_seq = [], []
            if INDEP_NUM == 2:
                for i, mlp_batch in enumerate(self.mlp_batch_embedded):
                    for j, input_batch in enumerate(self.input_batch_embedded):
                        coaction_sum, coaction_seq = gen_coaction(
                            mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j + 1)],
                            input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i + 1)], EMBEDDING_DIM,
                            mode=CALC_MODE, mask=self.mask)
                        tmp_sum.append(coaction_sum)
                        tmp_seq.append(coaction_seq)
            else:
                for i, (mlp_batch, input_batch) in enumerate(zip(self.mlp_batch_embedded, self.input_batch_embedded)):
                    coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM],
                                                              input_batch[:, :, : weight_emb_w[0][0]], EMBEDDING_DIM,
                                                              mode=CALC_MODE, mask=self.mask)
                    tmp_sum.append(coaction_sum)
                    tmp_seq.append(coaction_seq)

            self.coaction_sum = tf.concat(tmp_sum, axis=1)
            self.cross.append(self.coaction_sum)
        self.logger = set_logger()
        self.cl_loss = 0

    def attention_din_nomask_3dims(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
                                   din_deep_layers, din_activation, name_scope,
                                   att_type):
        # with tf.name_scope("attention_layer_%s" % (att_type)):
        # cur_poi_seq_fea_col [b, 18, eb]
        # hist_poi_seq_fea_col [b, 18, 10, eb]
        with tf.variable_scope("attention_layer_%s" % (att_type), reuse=tf.AUTO_REUSE):
            logger.info('attention layer')
            sess_num = cur_poi_seq_fea_col.get_shape().as_list()[1]  # 18 in [b,18,eb]
            embed_dim = cur_poi_seq_fea_col.get_shape().as_list()[-1]  # eb
            seq_len = hist_poi_seq_fea_col.get_shape().as_list()[-2]  # 10 in [b,18,10,eb]
            logger.info(
                '#3dims:seq_len {}; sess_num:{}; cur_poi_seq_fea_col:{}'.format(seq_len, sess_num, cur_poi_seq_fea_col))
            cur_poi_emb_rep = tf.tile(tf.reshape(cur_poi_seq_fea_col, [-1, sess_num, 1, embed_dim]),
                                      [1, 1, seq_len, 1])  # [b,18,10,eb]
            # 将query复制 seq_len 次 None, seq_len, embed_dim
            if att_type.startswith('top40'):
                din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
                logger.info('#wsl: top20 din is {}'.format(din_all))
            elif att_type == 'click_sess_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
            elif att_type == 'order_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
            else:
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

            logger.info('din_all %s ', din_all)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all  # [b,18,10,2*eb]
            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             name=name_scope + 'f_%d_att' % (i))
                # , reuse=tf.AUTO_REUSE
                input_layer = deep_layer  # [b,18,10,32]

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None,
                                               name=name_scope + 'fout_att')  # [b,18,10,1]
            logger.info('din_output_layer %s', din_output_layer)

            weighted_outputs = din_output_layer * hist_poi_seq_fea_col
            return weighted_outputs, din_output_layer

    def attention_din_nomask(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
                             din_deep_layers, din_activation, name_scope,
                             att_type):
        with tf.name_scope("attention_layer_%s" % (att_type)):
            # cur_poi_seq_fea_col [b, eb]
            # hist_poi_seq_fea_col [b, 10, eb]
            self.logger.info('attention layer')
            embed_dim = cur_poi_seq_fea_col.get_shape().as_list()[-1]  # eb
            seq_len = hist_poi_seq_fea_col.get_shape().as_list()[-2]  # 10 in [b,10,eb]
            cur_poi_emb_rep = tf.tile(tf.reshape(cur_poi_seq_fea_col, [-1, 1, embed_dim]), [1, seq_len, 1])

            # 将query复制 seq_len 次 None, seq_len, embed_dim
            if att_type.startswith('top40'):
                din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
                self.logger.info('#wsl: top20 din is {}'.format(din_all))
            elif att_type == 'click_sess_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
            elif att_type == 'order_att':
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
            else:
                din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

            self.logger.info('din_all %s ', din_all)

            activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
            input_layer = din_all
            for i in range(len(din_deep_layers)):
                deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                             name=name_scope + 'f_%d_att' % (i))
                # , reuse=tf.AUTO_REUSE
                input_layer = deep_layer

            din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')  # b,10,1
            self.logger.info('din_output_layer %s', din_output_layer)

            weighted_outputs = din_output_layer * hist_poi_seq_fea_col
            return weighted_outputs, din_output_layer

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2 if self.use_softmax else 1, activation=None, name='f3')
        return dnn3

    def build_loss(self, inp, cl_emb=None):

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            if self.use_softmax:
                self.y_hat = tf.nn.softmax(inp) + 0.00000001
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            else:
                self.y_hat = tf.nn.sigmoid(inp)
                ctr_loss = - tf.reduce_mean(tf.concat([tf.log(self.y_hat + 0.00000001) * self.target_ph,
                                                       tf.log(1 - self.y_hat + 0.00000001) * (1 - self.target_ph)],
                                                      axis=1))
            self.loss = ctr_loss
            if cl_emb:
                sim_mat = tf.matmul(cl_emb[0], tf.transpose(cl_emb[1], [1, 0]))

                cl_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.cl_label, logits=sim_mat)
                self.cl_loss = tf.reduce_mean(cl_loss)
                # cl_loss = tf.cond(tf.train.get_global_step() <= tf.constant(100000, dtype=tf.int64),
                #                   lambda: tf.constant(0.0), lambda: cl_loss)
                self.loss += 0.1 * self.cl_loss
            # self.loss += aux_loss

            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            if self.use_softmax:
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            else:
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

    def train(self, sess, inps):
        loss, accuracy, cl_loss, _ = sess.run([self.loss, self.accuracy, self.cl_loss, self.optimizer], feed_dict={
            # uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, sess_mask, fin_mid_sess,
            # fin_cat_sess, target, sl, lr

            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cate_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cate_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.mid_sess_his: inps[6],
            self.cat_sess_his: inps[7],
            self.mid_sess_tgt: inps[8],
            self.cat_sess_tgt: inps[9],
            self.sess_mask: inps[10],
            self.fin_mid_sess: inps[11],
            self.fin_cat_sess: inps[12],
            self.target_ph: inps[13],
            self.seq_len_ph: inps[14],
            self.lr: inps[15],
            self.cl_label: np.eye(inps[0].shape[0])
            # self.carte_batch_ph: inps[11]
        })
        return loss, accuracy, cl_loss

    def calculate(self, sess, inps):
        probs, loss, accuracy, cl_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.cl_loss], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cate_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cate_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.mid_sess_his: inps[6],
            self.cat_sess_his: inps[7],
            self.mid_sess_tgt: inps[8],
            self.cat_sess_tgt: inps[9],
            self.sess_mask: inps[10],
            self.fin_mid_sess: inps[11],
            self.fin_cat_sess: inps[12],
            self.target_ph: inps[13],
            self.seq_len_ph: inps[14],
            self.cl_label: np.eye(inps[0].shape[0])
        })
        return probs, loss, accuracy, cl_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_softmax=True):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling, use_softmax=use_softmax)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea],
            -1)
        # Fully connected layer
        logit = self.build_fcn_net(inp, use_dice=True)
        self.build_loss(logit)


class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_coaction=False):
        super(Model_DIEN, self).__init__(n_uid, n_mid, n_cate,
                                         EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                         use_negsampling, use_coaction=use_coaction)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             final_state2] + self.cross, 1)
        prop = self.build_fcn_net(inp, use_dice=True)
        self.build_loss(prop)


class Model_DDPM(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_softmax=True):
        super(Model_DDPM, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE,
                                         use_negsampling, use_softmax=use_softmax)
        self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his)  # [b, 18, 10, eb]
        self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_his)
        self.mid_sess_his_eb += self.cat_sess_his_eb
        self.mid_sess_tgt_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_tgt)  # [b, 18, eb]
        self.cat_sess_tgt_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_tgt)
        self.mid_sess_tgt_eb += self.cat_sess_tgt_eb
        self.fin_mid_sess_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.fin_mid_sess)  # [b, 10, eb]
        self.fin_cat_sess_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.fin_cat_sess)
        self.fin_mid_sess_eb += self.fin_cat_sess_eb
        # Attention layer
        # uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, sess_mask, fin_mid_sess, fin_cat_sess
        # self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)

        session_len = 10
        session_num = 18
        with tf.name_scope('Attention_layer'):
            # 1. Pathway Enhance
            mid_sess_his_eb = tf.reshape(self.mid_sess_his_eb, [-1, session_len * EMBEDDING_DIM])
            mid_sess_his_eb_enhance = se_block(mid_sess_his_eb, EMBEDDING_DIM, 'Pathway_Enhance_hist_pre')
            mid_sess_his_eb_enhance = tf.reshape(mid_sess_his_eb_enhance, [-1, session_num, session_len, EMBEDDING_DIM])

            fin_mid_sess_eb = tf.reshape(self.fin_mid_sess_eb, [-1, session_len * EMBEDDING_DIM])
            fin_mid_sess_eb_enhance = se_block(fin_mid_sess_eb, EMBEDDING_DIM, 'Pathway_Enhance_tgt_pre')
            fin_mid_sess_eb_enhance = tf.reshape(fin_mid_sess_eb_enhance, [-1, session_len, EMBEDDING_DIM])

            # 2. pathway match
            fin_mid_sess_eb_pool = tf.reduce_mean(fin_mid_sess_eb_enhance, axis=-2, keep_dims=True)  # [B, 1, eb]
            mid_sess_his_eb_pool = tf.reduce_mean(mid_sess_his_eb_enhance, axis=-2)  # [B, 18, eb]
            sess_score = tf.matmul(fin_mid_sess_eb_pool, tf.transpose(mid_sess_his_eb_pool, [0, 2, 1]))  # [B, 1, 18]
            attention_output_0 = din_attention(tf.squeeze(fin_mid_sess_eb_pool, 1), mid_sess_his_eb_pool,
                                               ATTENTION_SIZE, self.sess_mask, name_scope='attention_output_0')
            att_fea_0 = tf.reduce_sum(attention_output_0, 1)
            attention_output_1 = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                               att_score=tf.squeeze(sess_score, 1), name_scope='attention_output1')
            att_fea_1 = tf.reduce_sum(attention_output_1, 1)

            k_cluster = 8
            user_cluster = tf.get_variable('user_cluster_centers',
                                           shape=[1, k_cluster, EMBEDDING_DIM],
                                           initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1),
                                           )
            hist_seq_h = mid_sess_his_eb_pool + self.mid_sess_tgt_eb  # [B, 30, 8]
            hist_seq_mlp = tf.layers.dense(hist_seq_h, EMBEDDING_DIM, activation=tf.nn.tanh, use_bias=True,
                                           kernel_initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1),
                                           bias_initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1),
                                           name='hist_seq_mlp')  # [B, 30, 8]
            hist_seq_w = tf.matmul(user_cluster, tf.transpose(hist_seq_mlp, [0, 2, 1]))  # [B, k, 30]

            pre_seq_h = fin_mid_sess_eb_pool  # [B, 1, 8]
            pre_seq_mlp = tf.layers.dense(pre_seq_h, EMBEDDING_DIM, activation=tf.nn.tanh, use_bias=True,
                                          kernel_initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1),
                                          bias_initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1),
                                          name='pre_seq_mlp')  # [B, 1, 8]
            pre_seq_w = tf.matmul(user_cluster, tf.transpose(pre_seq_mlp, [0, 2, 1]))  # [B, k, 1]

            cluster_w = tf.matmul(tf.transpose(pre_seq_w, (0, 2, 1)), hist_seq_w)  # [B, 1, 30]
            attention_output_2 = din_attention(self.item_eb, hist_seq_mlp, ATTENTION_SIZE, self.sess_mask,
                                               att_score=tf.squeeze(cluster_w, 1), name_scope='attention_output2')
            att_fea_2 = tf.reduce_sum(attention_output_2, 1)

            cluster_e = tf.matmul(hist_seq_w, hist_seq_h)  # [B, k, 8]
            e_norm = tf.norm(cluster_e, axis=-1, keep_dims=True)  # [B, k, 1]
            L_D = (1.0 / k_cluster / k_cluster) * tf.reduce_sum(
                tf.math.divide_no_nan(tf.matmul(cluster_e, tf.transpose(cluster_e, [0, 2, 1])),
                                      tf.matmul(e_norm, tf.transpose(e_norm, [0, 2, 1]))))

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             att_fea_0, att_fea_1, att_fea_2], -1)

        # Fully connected layer
        logit = self.build_fcn_net(inp, use_dice=True)
        self.build_loss(logit)


class Model_DBPMaN(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
                 use_softmax=True):
        super(Model_DBPMaN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE,
                                           use_negsampling, use_softmax=use_softmax)
        self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his)  # [1024, 18, 10, eb]
        self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_his)
        self.mid_sess_his_eb += self.cat_sess_his_eb
        self.mid_sess_tgt_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_tgt)  # [1024, 18, eb]
        self.cat_sess_tgt_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_tgt)
        self.mid_sess_tgt_eb += self.cat_sess_tgt_eb
        self.fin_mid_sess_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.fin_mid_sess)  # [1024, 10, eb]
        self.fin_cat_sess_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.fin_cat_sess)
        self.fin_mid_sess_eb += self.fin_cat_sess_eb
        # Attention layer
        # uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, sess_mask, fin_mid_sess, fin_cat_sess
        # self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)

        # session_len = 10
        # session_num = 18
        with tf.name_scope('DBPMaN_Model'):
            # generate mask sequence [b, 18, 10]
            cl_mask1 = tf.where(tf.random.uniform(tf.shape(self.mid_sess_his), minval=0, maxval=1) > 0.2,
                                tf.ones_like(self.mid_sess_his, tf.int32),
                                tf.zeros_like(self.mid_sess_his, tf.int32))
            cl_mask2 = tf.where(tf.random.uniform(tf.shape(self.mid_sess_his), minval=0, maxval=1) > 0.2,
                                tf.ones_like(self.mid_sess_his, tf.int32),
                                tf.zeros_like(self.mid_sess_his, tf.int32))

            mid_sess_his_eb_cl1 = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his * cl_mask1) \
                                  + self.cat_sess_his_eb
            mid_sess_his_eb_cl2 = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his * cl_mask2) \
                                  + self.cat_sess_his_eb

            # 1. Pathway Enhance Module & generate cL_loss
            # 1.1 history path enhance
            # [b, 18, 10 ,eb]
            mid_sess_his_eb_enhance, nclk_his_att_score = self.attention_din_nomask_3dims(
                tf.stop_gradient(self.mid_sess_tgt_eb),
                self.mid_sess_his_eb,
                [64, 32], 'relu',
                'pem_his_att',
                'din_mlp')

            mid_sess_his_eb_enhance_cl1, nclk_his_att_score_cl1 = self.attention_din_nomask_3dims(
                tf.stop_gradient(self.mid_sess_tgt_eb),
                mid_sess_his_eb_cl1,
                [64, 32], 'relu',
                'pem_his_att',
                'din_mlp')

            mid_sess_his_eb_enhance_cl2, nclk_his_att_score_cl2 = self.attention_din_nomask_3dims(
                tf.stop_gradient(self.mid_sess_tgt_eb),
                mid_sess_his_eb_cl2,
                [64, 32], 'relu',
                'pem_his_att',
                'din_mlp')

            logger.info(
                "# pem_his_att_output:{}; pem_his_att_score:{}".format(mid_sess_his_eb_enhance, nclk_his_att_score))

            mask_pw_his_router = self.pathway_router_simple(mid_sess_his_eb_enhance, [32], 10,
                                                            'his_router')  # b,18,10,1

            mask_pw_his_router_cl1 = self.pathway_router_simple(mid_sess_his_eb_enhance_cl1, [32], 10,
                                                                'his_router')  # b,18,10,1

            mask_pw_his_router_cl2 = self.pathway_router_simple(mid_sess_his_eb_enhance_cl2, [32], 10,
                                                                'his_router')  # b,18,10,1

            mid_sess_his_eb_enhance = tf.reduce_sum((mid_sess_his_eb_enhance * mask_pw_his_router), axis=-2)  # b,18,eb
            out_fea_0 = tf.reduce_mean(mid_sess_his_eb_enhance, axis=-2)

            emb_cl1 = tf.reshape(mid_sess_his_eb_enhance_cl1 * mask_pw_his_router_cl1, [-1, 18, 10, EMBEDDING_DIM])
            # b,18,10,eb
            emb_cl2 = tf.reshape(mid_sess_his_eb_enhance_cl2 * mask_pw_his_router_cl2, [-1, 18, 10, EMBEDDING_DIM])
            # emb_cl1 = tf.reduce_mean(emb_cl1, axis=-2)
            # emb_cl2 = tf.reduce_mean(emb_cl2, axis=-2)
            emb_cl1 = tf.nn.l2_normalize(emb_cl1, dim=3, epsilon=1e-10, name='nn_l2_norm_cl1')
            emb_cl2 = tf.nn.l2_normalize(emb_cl2, dim=3, epsilon=1e-10, name='nn_l2_norm_cl2')
            emb_cl1 = tf.reshape(emb_cl1, [-1, 18*10*EMBEDDING_DIM])
            emb_cl2 = tf.reshape(emb_cl2, [-1, 18*10*EMBEDDING_DIM])
            # 1.2 cur path enhance
            # fin_mid_sess_eb = tf.reshape(self.fin_mid_sess_eb, [-1, session_len * EMBEDDING_DIM])
            # fin_mid_sess_eb_enhance = se_block(fin_mid_sess_eb, EMBEDDING_DIM, 'Pathway_Enhance_tgt_pre')
            # fin_mid_sess_eb_enhance = tf.reshape(fin_mid_sess_eb_enhance, [-1, session_len, EMBEDDING_DIM])

            mid_sess_cur_eb_enhance, nclk_cur_att_score = self.attention_din_nomask(
                tf.stop_gradient(self.mid_batch_embedded),
                self.fin_mid_sess_eb,
                [64, 32], 'relu',
                'pem_cur_att', 'din_mlp')
            self.logger.info(
                "#ZJ pem_cur_att_output:{}; pem_cur_att_output:{}".format(nclk_cur_att_score, nclk_cur_att_score))
            mask_pw_cur_router = self.pathway_router_simple(mid_sess_cur_eb_enhance, [32], 10,
                                                            'cur_router')  # b,10,1
            mid_sess_cur_eb_enhance = tf.reduce_sum((mid_sess_cur_eb_enhance * mask_pw_cur_router), axis=-2)  # b,eb
            out_fea_1 = mid_sess_cur_eb_enhance

            # 2. Pathway Matching Module
            mid_sess_cur_eb_pool = tf.reshape(mid_sess_cur_eb_enhance, [-1, 1, EMBEDDING_DIM])  # [B, 1, eb]
            mid_sess_his_eb_pool = mid_sess_his_eb_enhance  # [B, 18, eb]

            sess_score = tf.matmul(mid_sess_cur_eb_pool, tf.transpose(mid_sess_his_eb_pool, [0, 2, 1]))  # [B, 1, 18]
            attention_output_0 = din_attention(tf.squeeze(mid_sess_cur_eb_pool, 1), mid_sess_his_eb_pool,
                                               ATTENTION_SIZE, self.sess_mask, name_scope='attention_output_0')

            att_fea_0 = tf.reduce_sum(attention_output_0, 1)
            attention_output_1 = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                               att_score=tf.squeeze(sess_score, 1), name_scope='attention_output1')
            att_fea_1 = tf.reduce_sum(attention_output_1, 1)

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
             att_fea_0, att_fea_1, out_fea_0, out_fea_1], -1)

        # Fully connected layer
        logit = self.build_fcn_net(inp, use_dice=True)
        self.build_loss(logit, cl_emb=[emb_cl1, emb_cl2])

    def pathway_router_simple(self, input_layer, hidden_units_list, output_dim, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            net = input_layer  # b,18,10,eb or b,10,eb

            # net_reshape = tf.reduce_mean(net, axis=-1)  # b,18,10
            net_reshape = tf.squeeze(tf.concat(tf.split(net, net.shape[-2].value, axis=-2), axis=-1),
                                     axis=-2)  # b,18,10*eb
            # second loop
            for i in range(len(hidden_units_list)):
                net_reshape = tf.layers.dense(inputs=net_reshape,
                                              units=hidden_units_list[i],
                                              activation=tf.nn.relu,
                                              kernel_initializer=tf.glorot_normal_initializer(),
                                              bias_initializer=tf.glorot_normal_initializer(),
                                              name='%s_fc_pw_%d' % (name, i))
                self.logger.info(
                    "##gate_{}; input_layer:{}, gateNet:{},gate_units_list:{}".format(name, input_layer, net_reshape,
                                                                                      hidden_units_list))

            pw_router = tf.layers.dense(inputs=net_reshape,
                                        units=output_dim,
                                        activation=tf.nn.softmax,
                                        # activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.glorot_normal_initializer(),
                                        bias_initializer=tf.glorot_normal_initializer(),
                                        name='%s_softmax_pw' % (name))  # b,18,10

            self.logger.info("#ZJ name:{}; pw_router:{}".format(name, pw_router))
            topk_vals, _ = tf.nn.top_k(pw_router, 5)  # B,18,5
            min_topk_vals = tf.reduce_min(topk_vals, axis=-1, keepdims=True)  # B,18,1
            self.logger.info("#ZJ topk_vals:{}; min_topk_vals:{}".format(topk_vals,
                                                                         min_topk_vals))

            pw_bool = tf.math.greater_equal(pw_router, min_topk_vals)  # B, 18, 10
            mask_val = tf.zeros_like(pw_router)
            mask_pw_router = tf.where(pw_bool, pw_router, mask_val)  # b,18,10
            self.logger.info("#ZJ mask_val:{}; mask_pw_router:{}".format(mask_val,
                                                                         mask_pw_router))
            # return tf.reshape(net, [-1,1, output_dim])
        return tf.expand_dims(mask_pw_router, axis=-1)  # b,18,10,1
