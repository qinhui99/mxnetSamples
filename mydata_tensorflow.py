"""
[Author] hui qin
[E-mail] qinhui99@hotmail.com
[Create] 2017/04
[Last Modified] 2017/04/04
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.contrib import learn
import numpy as np


# 10000,3,d30,d45,d60
def loadDatas():
    with open("datas.txt", "r") as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])

        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int32)
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.int)
            target[i] = np.asarray(ir[-1], dtype=np.int32)
        return data, target


def main(unused_argv):
    qhdatas, qh_target = loadDatas()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        qhdatas, qh_target, test_size=0.2, random_state=42)

    # It's useful to scale to ensure Stochastic Gradient Descent
    # will do the right thing.
    scaler = StandardScaler()

    # DNN classifier.
    classifier = learn.DNNClassifier(
        feature_columns=learn.infer_real_valued_columns_from_input(x_train),
        hidden_units=[20, 10], n_classes=3)

    pipeline = Pipeline([('scaler', scaler),
                         ('DNNclassifier', classifier)])

    pipeline.fit(x_train, y_train, DNNclassifier__steps=20)

    score = accuracy_score(y_test, list(pipeline.predict(x_test)))
    print('Accuracy: {0:f}'.format(score))

    # the label is :[0,0,1,2,0,2,1,1,2,0]
    original_labels = np.array([0, 0, 1, 2, 0, 2, 1, 1, 2, 0])
    new_samples = np.array(
        [[14, 5, 30], [24, 5, 30], [4, 15, 45], [4, 15, 60], [23, 5, 30],
         [14, 5, 60], [24, 5, 45], [4, 15, 45], [4, 15, 60], [23, 5, 30]
         ],
        dtype=int)
    output = list(pipeline.predict(new_samples))

    # We get the [0,0,2,2,0,2,1,2,2,0]. The new samples accuracy is 0.6
    score = accuracy_score(original_labels, list(output))
    print('New samples accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()
