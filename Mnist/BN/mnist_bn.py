# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:15:36 2017

@author: minjiang
针对mnist手写体识别例子，比较使用BatchNormalization和不使用BN时深度网络的性能
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义层
def add_layer(inputs, in_size, out_size, keep_prob=1.0, batch_norm=False, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    
    score = tf.matmul(inputs, Weights)
    
    
    if not batch_norm:
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        score = score + biases
        
    else:
        batch_mean, batch_var = tf.nn.moments(score, [0])
        scale = tf.Variable(tf.ones([out_size]))
        beta = tf.Variable(tf.zeros([out_size]))
        
        score = tf.nn.batch_normalization(score, batch_mean, batch_var, beta, scale, BN_EPSILON)
        
    if activation_function:
        score = activation_function(score)
        
    return tf.nn.dropout(score, keep_prob)

#定义相关参数
BATCH_SIZE = 500
TRAINING_ITER = 3000
LEARNING_RATE = 0.1
HIDDEN_UNITS = [100, 100]
KEEP_PROB = 1.0
INPUT_UNITS = 784
N_CLASSES = 10
BN_EPSILON = 1e-3

#建立网络结构
x = tf.placeholder(tf.float32, [None, INPUT_UNITS])
y = tf.placeholder(tf.float32, [None, N_CLASSES])

h1 = add_layer(x, INPUT_UNITS, HIDDEN_UNITS[0], activation_function=tf.nn.sigmoid)
h1_bn = add_layer(x, INPUT_UNITS, HIDDEN_UNITS[0], batch_norm=True, activation_function=tf.nn.sigmoid)

h2 = add_layer(h1, HIDDEN_UNITS[0], HIDDEN_UNITS[1], activation_function=tf.nn.sigmoid)
h2_bn = add_layer(h1_bn, HIDDEN_UNITS[0], HIDDEN_UNITS[1], batch_norm=True, activation_function=tf.nn.sigmoid)

predictions = add_layer(h2, HIDDEN_UNITS[1], N_CLASSES)
predictions_bn = add_layer(h2_bn, HIDDEN_UNITS[1], N_CLASSES)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
loss_bn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions_bn))

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

train_step = optimizer.minimize(loss)
train_step_bn = optimizer.minimize(loss_bn)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predictions, 1), tf.arg_max(y, 1)), tf.float32))
accuracy_bn = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predictions_bn, 1), tf.arg_max(y, 1)), tf.float32))

#开始训练
init = tf.global_variables_initializer()

#不使用BN
valid_acc = []
epoch = []
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(TRAINING_ITER):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict = {x: batch_xs, y:batch_ys})
        if i % 50 == 0:
            epoch.append(i)
            print ('TRAIN_ACC@%d:' %i, sess.run(accuracy, feed_dict = {x: batch_xs, y: batch_ys}))
            valid_acc.append(sess.run(accuracy, feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}))
plt.title('MNIST_MLN_BN')
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
no_bn_line, = plt.plot(epoch, valid_acc, 'r-')

#使用BN
valid_acc = []

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(TRAINING_ITER):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step_bn, feed_dict = {x: batch_xs, y:batch_ys})
        if i % 50 == 0:
            print ('TRAIN_BN@%d:' %i, sess.run(accuracy_bn, feed_dict = {x: batch_xs, y: batch_ys}))
            valid_acc.append(sess.run(accuracy_bn, feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}))
bn_line, = plt.plot(epoch, valid_acc, 'y-')
plt.legend([no_bn_line, bn_line], ['Without BN', 'With BN'])
plt.ylim([0.8, 1.0])
plt.show()
