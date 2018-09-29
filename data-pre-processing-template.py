# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:10:27 2018

@author: anusha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv');
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.2, random_state = 0)

"""sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)"""