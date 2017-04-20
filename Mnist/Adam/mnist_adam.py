#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:25:52 2017

@author: minjiang
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

#权重初始化采用截断正态分布（2 sigma以内）
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)

#偏置初始化为0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)

#卷积函数定义，strides：卷积模板移动步长，都是1代表会不遗漏地划过图片中的每一个点
#SAME：代表给边界加上Padding让卷积的输出和输入保存一致
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化函数定义，ksize： 使用2*2的池化窗口；strides：相邻池化窗口的水平/竖直位移，2*2
#此时为无重叠池化
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#定义迭代次数；batch的大小；学习率
epoch_max = 20000
batch_size = 100
Learning_rate = 1e-4

#将1维输入向量（1*784）转为2为（28*28）；-1代表样本数不固定
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

#定义第一个卷积层，卷积核大小5*5，1个颜色通道，卷积核数目32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#使用tanh激励函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#定义第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#定义第一个全连接层，隐层节点数为1024
#两次池化之后，图片尺寸由28*28变为7*7，因此一维输出的尺寸为7*7*64
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#定义输出层，softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

#定义损失函数，交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#采用Adam来优化，学习率为1E-4
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(cross_entropy)

#定义评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#开始训练
sess.run(tf.global_variables_initializer())

epoch = []
train_acc = []
valid_acc = []

for i in range(epoch_max):
  batch = mnist.train.next_batch(batch_size)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  if i%100 == 0:
      epoch.append(i)
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
      train_acc.append(train_accuracy)
      
 #     valid_accuracy = accuracy.eval(feed_dict={x:mnist.validation.images, y_:mnist.validation.labels})
 #     valid_acc.append(valid_accuracy)


#test_acc  = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})

#逐批次地对测试样本进行预测
test_num = np.shape(mnist.test.images)[0]
test_epoch = test_num // batch_size
test_acc = []
for i in range(test_epoch):
    test_accuarcy  = accuracy.eval(feed_dict={x:mnist.test.images[i*batch_size:(i+1)*batch_size],
                             y_:mnist.test.labels[i*batch_size:(i+1)*batch_size]})
    test_acc.append(test_accuarcy)
test_acc_mean = np.mean(test_acc)

np.save('train_acc_adam.npy', train_acc)
np.save('test_acc_adam.npy', test_acc_mean)
#画图,和使用随机梯度下降法的结果比较
train_acc_relu = np.load('train_acc_relu.npy')
plt.plot(epoch, train_acc, 'r--', label='Adam')
plt.plot(epoch, train_acc_relu, 'b--', label='Sgd')
plt.title('The change of accuracy as the epoch increases with different optimization methods')
plt.ylabel('Accuracy')
plt.xlabel('epcoh')
plt.legend()
plt.show()
