#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:22:39 2017

@author: minjiang
分别使用决策树，随机森林，梯度提升树对泰坦尼克号乘客是否生还进行预测
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#读入数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#查看数据的统计特征，可以发现数据存在缺失
#titanic.info()

#人工选取pclass, age和sex作为判别是否生还的特征
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

#查看当前选取特征的数据情况,发现age列需要补充
#X.info()
#使用平均数补充
X['age'].fillna(X['age'].mean(), inplace=True)

#对原始数据进行分割，25%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

#对类别特征进行转换，成为特征向量
vec = DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

#使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)
print('The accuracy of decision tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

#使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)
print('The accuracy of random forest is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

#使用梯度提升决策树进行集成模型的训练以及预测分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)
print('The accuracy of gradient tree boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))
