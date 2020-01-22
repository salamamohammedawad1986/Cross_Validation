#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:17:48 2019

@author: salama
"""

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns

df = pd.read_csv('nona1.csv')
target = df.code

df.head()
df.total = df.total.fillna(df.total.mean())

label_X = LabelEncoder()
sc = StandardScaler()

df['code'] = label_X.fit_transform(df['code'])

x,y = df.drop(['code'], axis=1).values, df['code'].values

sc_x = sc.fit_transform(x)

xtrain,xtest,ytrian,ytest = train_test_split(sc_x,y, test_size=0.2,random_state=1)

clf = SVC(kernel='linear')

clf.fit(xtrain,ytrian)

pred = clf.predict(xtest)

def models():
    model = []
    model.append(('THIS ACCURY :{}'.format(accuracy_score(ytest, pred))))
    model.append(('CONFUSION : {}'.format(confusion_matrix(ytest, pred))))
    model.append(('RECALL : {}'.format(recall_score(ytest, pred))))
    model.append(('CLASSFICATION : {}'.format(classification_report(ytest,pred))))
    for i in model:
        print(i)
models()        


mat = confusion_matrix(ytest, pred)

sns.heatmap(mat, square=True,annot=True, fmt='d',cbar=True,
            xticklabels=x[0:],
            yticklabels=x[0:])

#===================================================================#
plt.scatter(x[0:])

clf_new = SVC(kernel='linear').fit(x, y)
clf_new.predict(x)

plt.scatter(x[:,0],x[:,1], c=y,s=50, cmap='autumn')
plt.plot(x, clf.predict(x))











