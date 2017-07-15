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

Mnist/BN/:针对MNIST数据集，比较分别使用和不使用BN操作时神经网络的性能。

Tree_Titanic：针对titanic数据集，分别使用决策树，随机森林，梯度提升树对泰坦尼克号乘客是否生还进行预测，并对三种算法进行了特性选取的探究。

svm_gridsearch.py: 针对手写体识别例子，使用支持向量机模型来分类，采用高斯核。并进行超参数的搜索（gamma, C），采用网格化搜索。

SNN_CNN_MNIST.py：针对手写体识别例子，采用卷积神经网络，比较分别使用relu和selu激励函数的效果。

WGan_Mnist.py：针对手写体识别例子，实现WGan_gp生成对抗网络。
