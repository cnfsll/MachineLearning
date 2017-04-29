# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:04:28 2017

@author: minjiang
针对手写体识别例子，使用支持向量机模型来分类，采用高斯核
并进行超参数的搜索（gamma, C），采用网格化搜索
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd

#读取数据集
digits = load_digits()

#分割数据集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, random_state=33)

#参数范围设置
tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-4,1,6),
                     'C': np.logspace(-1,4,6)}]

#执行网格化搜索
gs = GridSearchCV(SVC(), tuned_parameters, cv=5)
gs.fit(X_train, y_train)

#显示最佳参数
print("Best parameters set found on development set:")
print()
print(gs.best_params_)

#显示每组参数的结果
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

#输出最佳参数在测试集上的分类结果
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, gs.predict(X_test)
print(classification_report(y_true, y_pred))
print()