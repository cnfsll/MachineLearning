# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:20:38 2017

@author: minjiang
"""

"""卷积神经网络，基于LeNeT5
"""
import timeit
import cPickle
import gzip

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
 

def load_data(dataset):
    #dataset是数据集的路径
    #从"minst.pkl.gz"里加载train_set,vaild_set,test_set
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    #将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中
    #GPU里数据类型只能是floatX。而data_y是类别，所以最后又转换为int返回
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)]
    return rval

#定义卷积层
class LeNetConvPoolLayer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        
        #input：theano.tensor.dtensor4
        #filter_shape：4维元组或者列表(滤波器数目 输入特征图的个数 滤波器的高度 宽度)
        #image_shape: 4维元组或者列表(batch大小 输入特征图的个数 图像的高度 宽度)
        #assert condition 当条件为真，继续执行，否则中断程序
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        #每个隐层神经元与上一层的连接数为 输入特征图的个数*滤波器的高度*滤波器的宽度
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # 初始化权重
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        #初始化偏置
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # 卷积操作
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        #池化操作，maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        #加偏置，再tanh映射
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        #存储参数
        self.params = [self.W, self.b]

        self.input = input

#定义隐层层
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        rng：初始化权重
        input：(n_examples, n_in)
        n_in：输入数目
        n_out：输出数目（隐层节点数目）
        activation:激活函数，这里取tanh
        """
        self.input = input
    

        #W初始化规则：如果使用tanh函数，则在-sqrt(6./(n_in+n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀
        #抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍
 
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

#定义分类层
class LogisticRegression(object):
    """多类LogisticRegression，即softmax回归
    """

    def __init__(self, input, n_in, n_out):
        """ 
        input:(n_example,n_in)，其中n_example是一个batch的大小
        n_in：输入数目，即上一隐层的输出
        n_out:输出的类比数
        """
        
        # 初始化w
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # 初始化偏置
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # W的第k列表示类别k的超平面的参数
        # input每一行表示一个样本
        # b的第k个元素表示类别k的超平面的参数
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # 预测得到每个样本的类别
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        """负对数似然函数，作为代价函数
        """
       
        # T.log(self.p_y_given_x)是一个矩阵，行表示样本，类表示该样本属于某类别的概率
        # [T.arange(y.shape[0]), y]: [0,y(0);1,y(1);....;n-1,y(n-1)]
        # T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]即得到每个样本属于标签类(y)的概率
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """定义预测误差函数
        """

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            # 当预测类别和实际类别不同时，T.neq返回1
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

#卷积网络建立，训练
dataset='mnist.pkl.gz'
#参数设置
learning_rate = 0.1 #学习率
n_epochs = 20  #训练步数，每一步都会遍历所有的batch
nkerns = [20,50] #两个卷积层，滤波器的数目
batch_size = 500 #batch的大小

rng = numpy.random.RandomState(23455)

#加载数据
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# 计算batch的个数
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size
n_test_batches /= batch_size

#定义几个变量
index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

###############
#  建立模型  #
###############
print '建立模型'

# 将输入图像 (batch_size, 28 * 28)转变成一个4维数据
layer0_input = x.reshape((batch_size, 1, 28, 28))
#灰度图只有1个特征
# 建立一个卷积层
# 滤波后图像大小 (28-5+1 , 28-5+1) = (24, 24)
# 池化后 (24/2, 24/2) = (12, 12)
# 输出 (batch_size, nkerns[0], 12, 12)
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 28, 28),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2)
)

# 建立第一个卷积层
#滤波后图像大小 (12-5+1, 12-5+1) = (8, 8)
# 池化后 (8/2, 8/2) = (4, 4)
# 输出 (batch_size, nkerns[1], 4, 4)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 12, 12),
    filter_shape=(nkerns[1], nkerns[0], 5, 5),
    poolsize=(2, 2)
)

# 第二个卷积层的输出与一个隐含层全连接
# 需要将layer1的输出flatten为(batch_size,nkerns[1]*4*4)
# 输入[batch_size,500]
layer2_input = layer1.output.flatten(2)
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 4 * 4,
    n_out=500,
    activation=T.tanh
)

# 分类层
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

# 代价函数
cost = layer3.negative_log_likelihood(y)

# 计算测试误差
test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

#计算验证误差
validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# 参数列表
params = layer3.params + layer2.params + layer1.params + layer0.params

# 计算梯度
grads = T.grad(cost, params)

# 参数更新
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

# 训练模型
train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

###############
#   开始训练  #
###############
print('训练模型')

# 早停参数 提供一种早停实现策略
patience = 10000  
patience_increase = 2
improvement_threshold = 0.995  # 验证准确度提高比例
validation_frequency = min(n_train_batches, patience // 2)
#这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。

best_validation_loss = numpy.inf
best_iter = 0 #最好的迭代次数，以batch为单位
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

#while循环控制epoch
#for循环是遍历一个个batch

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
 
        iter = (epoch - 1) * n_train_batches + minibatch_index
        #cost_ij没有实际用处，只是为了调用train_model
        cost_ij = train_model(minibatch_index)
        #每一个epcoh结束后，在验证集上测试
        if (iter + 1) % validation_frequency == 0:

            # 计算验证集误差
            validation_losses = [validate_model(i) for i
                                 in range(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            # 如果验证误差小于之前的最佳误差，则更新best_validation_loss和best_iter
            if this_validation_loss < best_validation_loss:

                #如果验证集误差小于best_validation_loss*improvement_threshold
                #说明网络需要调整还不是很充分，以几何速度调整patience的值

                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # 计算测试误差
                test_losses = [
                    test_model(i)
                    for i in range(n_test_batches)
                ]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))
#patience可以表征网络调整的怎么样，当网络一直验证集上的错误率值一直都是很小幅度或者停止不进的更改
#那么这个值不会调整，它将很快超过minibatch的次数，那么就停止训练
        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print('The code run for %d epochs, with %f min' % (
    epoch,  (end_time - start_time) / 60))


