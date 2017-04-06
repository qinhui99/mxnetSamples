"""
[Author] hui qin
[E-mail] qinhui99@hotmail.com
[Create] 2017/04
[Last Modified] 2017/04/04
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import cross_validation
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.contrib import learn


def main(unused_argv):
  iris = load_iris()
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

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
  import numpy as np
  # the label is 
  #[1,0,2,1,0,1,1,1,2,1]
  original_labels = np.array([1,0,2,1,0,1,1,1,2,1])
  new_samples = np.array(
      [[5.5, 2.3, 4.0, 1.3], [5.0, 3.3, 1.4, 0.2], [5.8, 2.7, 5.1, 1.9], [5.5, 2.6, 4.4, 1.2], [4.9, 3.1, 1.5, 0.1],
       [4.9, 2.4, 3.3, 1.0], [6.0, 2.2, 4.0, 1.0], [6.6, 2.9, 4.6, 1.3], [6.4, 2.8, 5.6, 2.1], [5.5, 2.3, 4.0, 1.3]],
      dtype=float)

  output=list(pipeline.predict(new_samples))
  #We get the 
  score = accuracy_score(original_labels, list(output))
  print('New samples accuracy: {0:f}'.format(score))
  for a in output:
      print(a),

if __name__ == '__main__':
  tf.app.run()
