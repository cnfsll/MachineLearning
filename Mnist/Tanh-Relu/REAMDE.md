针对手写体识别（MNIST）的例子，比较分别使用tanh和relu激励函数的卷积神经网络的效果。
实验证明，使用relu时收敛更快，最终测试准确率更高（98.85% VS 98.79%)
mnist_tanh.py:使用tanh激励函数
mnist_rely.py:使用relu激励函数
mnist_tanh.png,train_acc_tanh.npy,test_acc_tanh.npy:使用tanh激励函数得到的实验结果
mnist_relu.png,train_acc_relu.npy,test_acc_relu.npy:使用relu
