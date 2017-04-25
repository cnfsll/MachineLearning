#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:24:58 2017

@author: minjiang
使用决策树对泰坦尼克号乘客是否生还进行预测
并使用特征筛选来寻找最佳的特征组合
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import feature_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

#读入数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#查看数据的统计特征，可以发现数据存在缺失
#titanic.info()

#分离数据特征与预测目标
X = titanic.drop(['row.names', 'name', 'survived'], axis = 1)
y = titanic['survived']

#对缺失数据进行填充
#X.info()
#使用平均数补充年龄
X['age'].fillna(X['age'].mean(), inplace=True)
#其他缺失数据用“UNKNOW”填充
X.fillna('UNKNOWN', inplace=True)

#对原始数据进行分割，25%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

#对类别特征进行转换，成为特征向量
vec = DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

#输出处理后特征向量的维度
print(len(vec.feature_names_))

#使用所有的特征
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)
print(dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

#筛选前20%的特征，使用相同配置的决策树模型
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dtc.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dtc_y_pred = dtc.predict(X_test_fs)
print(dtc.score(X_test_fs, y_test))
print(classification_report(dtc_y_pred, y_test))

#通过交叉验证的方法，按照固定间隔的百分比筛选特征
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dtc, X_train_fs, y_train, cv=5)
    results.append(scores.mean())
print (results)

#找到体现最佳性能的特征筛选的百分比
opt = results.index(max(results))
print('Optimal numbel of features %d' %percentiles[opt])

plt.plot(percentiles, results)
plt.xlabel('percentiles of features')
plt.ylabel('accuarcy')
plt.show()

#使用最佳筛选后的特征（7%），利用相同配置的决策树模型
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dtc.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dtc_y_pred = dtc.predict(X_test_fs)
print(dtc.score(X_test_fs, y_test))
print(classification_report(dtc_y_pred, y_test))
