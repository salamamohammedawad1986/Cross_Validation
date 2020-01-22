#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:34:57 2020

@author: salama
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import load_digits
from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)

# import warnings filterfrom warnings import simplefilter


dg = load_digits()
xtrain,xtest,ytrain,ytest = train_test_split(dg.data,dg.target, test_size=0.3)


def getmodel(model_S, xtrain,xtest,ytrain,ytest):
    model_S.fit(xtrain,ytrain)
    return model_S.score(xtest, ytest)

cl_rf = []
cl_cv = []
cl_lo = []
fk = KFold(n_splits=3)
for train_index, test_index in fk.split(dg.data):
    xtrain,xtest,ytrain,ytest = dg.data[train_index], dg.data[test_index], \
          dg.target[train_index],dg.target[test_index]  
    cl_rf.append(getmodel(RandomForestClassifier(n_estimators=40), xtrain,xtest,ytrain,ytest))
    cl_cv.append(getmodel(SVC(), xtrain,xtest,ytrain,ytest))
    cl_lo.append(getmodel(LogisticRegression(), xtrain,xtest,ytrain,ytest))
    
    
#--------------------------------------------------------------------------------------
#THIS SCORE MEASUREMENT 
ncl_rf = []
ncl_cv = []
ncl_lo = []
from sklearn.model_selection import cross_val_score

ncl_lo.append(cross_val_score(LogisticRegression(), dg.data,dg.target))
ncl_cv.append(cross_val_score(SVC(), dg.data,dg.target))
ncl_rf.append(cross_val_score(RandomForestClassifier(n_estimators=40), dg.data,dg.target))
                                        