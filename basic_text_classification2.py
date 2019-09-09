from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print('Training entries: {}, labels: {}'.format(len(train_data), len(train_labels)))

print(train_data[0])

print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}

word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
