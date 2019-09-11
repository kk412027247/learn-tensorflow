from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
print(dftrain.describe())

print(dftrain.shape[0], dfeval.shape[0])

plt.figure()
dftrain.age.hist(bins=20)
plt.show()

plt.figure()
dftrain.sex.value_counts().plot(kind='barh')
plt.show()

plt.figure()
dftrain['class'].value_counts().plot(kind='barh')
plt.show()

plt.figure()
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()
