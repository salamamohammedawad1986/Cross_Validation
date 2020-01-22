#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:06:18 2020

@author: salama
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()

xtrain,xtest,ytrain,ytest = train_test_split(digits.data,digits.target, test_size =0.3)


#YOU CAN USE ALL MEDTHD
lr = LogisticRegression()
lr.fit(xtrain, ytrain)
lr.score(xtest,ytest)

#-----------------------------------------------------
svm1 = SVC()
svm1.fit(xtrain, ytrain)
svm1.score(xtest, ytest)

#--------------------------------------------------------
rdm = RandomForestClassifier(n_estimators=40)
rdm.fit(xtrain, ytrain)
rdm.score(xtest, ytest)
#---------------------------------------------------
#SULATION FOR THIS PROBLEM USE KFold
from sklearn.model_selection import KFold

kf = KFold(n_splits=3)

for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)




