# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:33:48 2021

@author: nidhi
"""

#import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

data = pd.read_csv('D:/projects/winequality-red.csv')

data['quality'] = data['quality'].map({3 : 'bad', 4 :'bad', 5: 'bad',
                                      6: 'good', 7: 'good', 8: 'good'})


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['quality'] = le.fit_transform(data['quality'])

data['quality'].value_counts


x = data.iloc[:,:11]
y = data.iloc[:,11]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier

# creating the model
model = RandomForestClassifier(n_estimators = 1000)

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the results for the test set
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix = \n{cm}")

ac = accuracy_score(y_test, y_pred)
print(f"Accuracy = {ac*100}%")