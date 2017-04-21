# MachineLearning
To record my learning process of machine learning

mycnn.py：基于python theano 实现卷积神经网络
参考如下：
http://deeplearning.net/tutorial/
- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

vgg16.py:
利用tensorflow搭建VGGNet16网络结构，并评测其inference耗时和training耗时
参考如下：
https://github.com/machrisaa/tensorflow-vgg

Mnist/Tanh-Relu/：针对MNIST数据集，比较分别使用tanh和relu激励函数的卷积神经网络的性能。

Mnist/Adam/: 针对MNIST数据集，比较分别使用随机梯度下降法和Adam优化方法时卷积神经网络的性能。

Mnist/Dropout/: 针对MNIST数据集，比较分别使用和不使用dropout操作时卷积神经网络的性能。
