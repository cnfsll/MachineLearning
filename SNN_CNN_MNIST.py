#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:27:08 2017

@author: minjiang
卷积网络结构
SNN结构
SELUs激励函数
参考地址：https://github.com/bioinf-jku/SNNs
"""
import tensorflow as tf
import numpy as np


import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

from sklearn.preprocessing import StandardScaler
from scipy.special import erf,erfc

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义selu函数
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

#定义适应selu的dropout技术
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):


    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

#将输入归一化
scaler = StandardScaler().fit(mnist.train.images)

#参数设置
learning_rate = 0.025
training_iters = 100
batch_size = 128
display_step = 1

#网络参数
n_input = 784
n_classes = 10
keep_prob_ReLU = 0.5 #对于relu的dropout率
dropout_prob_SNN = 0.05 # 对于selu的dropout律


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #对于relu的dropout率
dropout_prob =  tf.placeholder(tf.float32) #对于selu的dropout律
is_training = tf.placeholder(tf.bool)

#定义relu和selu的卷积操作和池化操作
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv2d_SNN(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return selu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


#创建模型 relu
def conv_net_ReLU(x, weights, biases, keep_prob):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    fc1 = tf.nn.dropout(fc1, keep_prob)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

#创建模型 selu
def conv_net_SNN(x, weights, biases, dropout_prob, is_training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d_SNN(x, weights['wc1'], biases['bc1'],)
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d_SNN(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = selu(fc1)
    
    fc1 = dropout_selu(fc1, dropout_prob,training=is_training)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

#relu权重和偏置
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32],stddev=np.sqrt(2/25)) ),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64],stddev=np.sqrt(2/(25*32)))),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024],stddev=np.sqrt(2/(7*7*64)))),
    'out': tf.Variable(tf.random_normal([1024, n_classes],stddev=np.sqrt(2/(1024))))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32],stddev=0)),
    'bc2': tf.Variable(tf.random_normal([64],stddev=0)),
    'bd1': tf.Variable(tf.random_normal([1024],stddev=0)),
    'out': tf.Variable(tf.random_normal([n_classes],stddev=0))
}

#selu权重和偏置
weights2 = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32],stddev=np.sqrt(1/25)) ),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64],stddev=np.sqrt(1/(25*32)))),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024],stddev=np.sqrt(1/(7*7*64)))),
    'out': tf.Variable(tf.random_normal([1024, n_classes],stddev=np.sqrt(1/(1024))))
}

biases2 = {
    'bc1': tf.Variable(tf.random_normal([32],stddev=0)),
    'bc2': tf.Variable(tf.random_normal([64],stddev=0)),
    'bd1': tf.Variable(tf.random_normal([1024],stddev=0)),
    'out': tf.Variable(tf.random_normal([n_classes],stddev=0))
}


#模型构建
pred_ReLU = conv_net_ReLU(x, weights, biases, keep_prob)
pred_SNN = conv_net_SNN(x, weights2, biases2, dropout_prob,is_training)

#定义损失函数和优化器
cost_ReLU = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_ReLU, labels=y))
cost_SNN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_SNN, labels=y))

optimizer_ReLU = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_ReLU)
optimizer_SNN = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_SNN)

#模型测试
correct_pred_ReLU = tf.equal(tf.argmax(pred_ReLU, 1), tf.argmax(y, 1))
accuracy_ReLU = tf.reduce_mean(tf.cast(correct_pred_ReLU, tf.float32))

#计算准确率
correct_pred_SNN = tf.equal(tf.argmax(pred_SNN, 1), tf.argmax(y, 1))
accuracy_SNN = tf.reduce_mean(tf.cast(correct_pred_SNN, tf.float32))

init = tf.global_variables_initializer()

training_loss_protocol_ReLU = []
training_loss_protocol_SNN = []


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.Session(config=config) as sess:
    sess.run(init)
    step = 0

    while step < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x_norm = scaler.transform(batch_x)

        sess.run(optimizer_ReLU, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: keep_prob_ReLU})
        sess.run(optimizer_SNN, feed_dict={x: batch_x_norm, y: batch_y,
                                       dropout_prob: dropout_prob_SNN,is_training:True})
        
        
        if step % display_step == 0:

            loss_ReLU, acc_ReLU = sess.run([cost_ReLU, accuracy_ReLU], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.0})
            training_loss_protocol_ReLU.append(loss_ReLU)
            
            loss_SNN, acc_SNN = sess.run([cost_SNN, accuracy_SNN], feed_dict={x: batch_x_norm,
                                                              y: batch_y,
                                                              dropout_prob: 0.0, is_training:False})
            training_loss_protocol_SNN.append(loss_SNN)
            
            print( "RELU: Nbr of updates: " + str(step+1) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss_ReLU) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc_ReLU))
            
            print( "SNN: Nbr of updates: " + str(step+1) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss_SNN) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc_SNN))
        step += 1
    print("Optimization Finished!\n")

    print("ReLU: Testing Accuracy:", sess.run(accuracy_ReLU, feed_dict={x: mnist.test.images[:512],
                                      y: mnist.test.labels[:512],
                                      keep_prob: 1.0}))
    print("SNN: Testing Accuracy:", sess.run(accuracy_SNN, feed_dict={x: scaler.transform(mnist.test.images[:512]),
                                      y: mnist.test.labels[:512],
                                      dropout_prob: 0.0, is_training:False}))


import matplotlib.pyplot as plt
#画图
fig, ax = plt.subplots()
ax.plot( training_loss_protocol_ReLU, label='Loss ReLU-CNN')
ax.plot( training_loss_protocol_SNN, label='Loss SNN')
ax.set_yscale('log')  # log scale
ax.set_xlabel('iterations/updates')
ax.set_ylabel('training loss')
fig.tight_layout()
ax.legend()
fig