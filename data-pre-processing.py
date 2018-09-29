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

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', stratergy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transfrom(x[:, 1:3])



