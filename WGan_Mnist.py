#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:36:51 2017

@author: minjiang
在MNIST数据集上训练WGAN_GP
参考如下
https://github.com/Zardinality/WGAN-tensorflow
https://zhuanlan.zhihu.com/p/25071913?utm_source=weibo&utm_medium=social

"""

from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.examples.tutorials.mnist import input_data


#定义leaky Relu激活函数
def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

#参数设置
batch_size = 64
z_dim = 128
learning_rate_ger = 5e-5
learning_rate_dis = 5e-5
device = '/cpu:0'
s = 32    #图像大小
Citers = 5    #判别器更新5次，生成器更新1次
max_iter_step = 20000
#梯度惩罚系数
lam = 10.


s2, s4, s8, s16 =\
    int(s / 2), int(s / 4), int(s / 8), int(s / 16)

# 储存运行日志，损失值的地址
log_dir = './log_wgan'
ckpt_dir = './ckpt_wgan'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


#生成器，卷积
def generator_conv(z):
    train = ly.fully_connected(
        z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4, 512))
    #从4*4变为8*8,stride=2
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', 
                                weights_initializer=tf.random_normal_initializer(0, 0.02),
                                normalizer_params={'is_training':True})
    #从8*8变为16*16
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', 
                                weights_initializer=tf.random_normal_initializer(0, 0.02), 
                                normalizer_params={'is_training':True})
    #从16*16变为32*32
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', 
                                weights_initializer=tf.random_normal_initializer(0, 0.02),
                                normalizer_params={'is_training':True})
    train = ly.conv2d_transpose(train, 1, 3, stride=1, normalizer_fn=ly.batch_norm,
                                activation_fn=tf.nn.tanh, padding='SAME', 
                                weights_initializer=tf.random_normal_initializer(0, 0.02),
                                normalizer_params={'is_training':True})
    return train


#判别器，卷积
#判别器不使用批标准化
def critic_conv(img, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        img = ly.conv2d(img, num_outputs=size, kernel_size=3,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
                        stride=2, activation_fn=lrelu,
                        normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
                        stride=2, activation_fn=lrelu, 
                        normalizer_params={'is_training':True})

        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3,
                        stride=2, activation_fn=lrelu,
                        normalizer_params={'is_training':True})
        logit = ly.fully_connected(tf.reshape(
            img, [batch_size, -1]), 1, activation_fn=None)
    return logit


def build_graph():

    noise_dist = tf.contrib.distributions.Normal(0., 1.)
    z = noise_dist.sample((batch_size, z_dim))
    generator = generator_conv
    critic = critic_conv
    with tf.variable_scope('generator'):
        train = generator(z)
    real_data = tf.placeholder(
        dtype=tf.float32, shape=(batch_size, 32, 32, 1))
    true_logit = critic(real_data)
    fake_logit = critic(train, reuse=True)
    #判别器损失函数
    c_loss = tf.reduce_mean(fake_logit - true_logit)
    #最终的损失函数，加上梯度惩罚
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((batch_size, 1, 1, 1))
    interpolated = real_data + alpha*(train-real_data)
    inte_logit = critic(interpolated, reuse=True)
    gradients = tf.gradients(inte_logit, [interpolated,])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
    
    #记录和汇总
    gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
    grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
    
    c_loss += lam*gradient_penalty
    
    #生成器损失函数
    g_loss = tf.reduce_mean(-fake_logit)
    #记录和汇总
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    img_sum = tf.summary.image("img", train, max_outputs=10)
    
    #生成器参数
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #判别器参数
    theta_c = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    #采用RMSProp优化方法
    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
                    optimizer= tf.train.RMSPropOptimizer, 
                    variables=theta_g, global_step=counter_g,
                    summaries = ['gradient_norm'])
    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
                    optimizer= tf.train.RMSPropOptimizer, 
                    variables=theta_c, global_step=counter_c,
                    summaries = ['gradient_norm'])

    return opt_g, opt_c, real_data

def main():
    #读取MNIST数据集
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    with tf.device(device):
        opt_g, opt_c, real_data = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    def next_feed_dict():
        train_img = dataset.train.next_batch(batch_size)[0]
        train_img = 2*train_img-1

        train_img = np.reshape(train_img, (-1, 28, 28))
        npad = ((0, 0), (2, 2), (2, 2))
        train_img = np.pad(train_img, pad_width=npad,
                           mode='constant', constant_values=-1)
        train_img = np.expand_dims(train_img, -1)
        feed_dict = {real_data: train_img}
        return feed_dict
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        for i in range(max_iter_step):
            if i < 25 or i % 500 == 0:
                citers = 100
            else:
                citers = Citers
            #训练判别器
            for j in range(citers):
                feed_dict = next_feed_dict()
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    sess.run(opt_c, feed_dict=feed_dict)                
            feed_dict = next_feed_dict()
            #训练生成器
            if i % 100 == 99:
                _, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata {}'.format(i), i)
            else:
                sess.run(opt_g, feed_dict=feed_dict)                
            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)
                print(opt_c)

main()
