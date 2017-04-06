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
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.contrib import learn

#208,60,M,R
def loadDatas():
    with open("sonar_number.txt","r") as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)
        return data,target

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
      hidden_units=[70, 10], n_classes=2)

    pipeline = Pipeline([('scaler', scaler),
                       ('DNNclassifier', classifier)])

    pipeline.fit(x_train, y_train, DNNclassifier__steps=20)

    score = accuracy_score(y_test, list(pipeline.predict(x_test)))
    print('Accuracy: {0:f}'.format(score))
    import numpy as np
    # the label is :[0,1,0,1]
    original_labels = np.array([0,1,0,1])
    new_samples = np.array(
        [[0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111, 0.1609, 0.1582, 0.2238,
          0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415,
          0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943,
          0.2744, 0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324, 0.0232, 0.0027,
          0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032],
         [0.0629, 0.1065, 0.1526, 0.1229, 0.1437, 0.1190, 0.0884, 0.0907, 0.2107, 0.3597, 0.5466, 0.5205, 0.5127,
          0.5395, 0.6558, 0.8705, 0.9786, 0.9335, 0.7917, 0.7383, 0.6908, 0.3850, 0.0671, 0.0502, 0.2717, 0.2839,
          0.2234, 0.1911, 0.0408, 0.2531, 0.1979, 0.1891, 0.2433, 0.1956, 0.2667, 0.1340, 0.1073, 0.2023, 0.1794,
          0.0227, 0.1313, 0.1775, 0.1549, 0.1626, 0.0708, 0.0129, 0.0795, 0.0762, 0.0117, 0.0061, 0.0257, 0.0089,
          0.0262, 0.0108, 0.0138, 0.0187, 0.0230, 0.0057, 0.0113, 0.0131],
         [0.0181, 0.0146, 0.0026, 0.0141, 0.0421, 0.0473, 0.0361, 0.0741, 0.1398, 0.1045, 0.0904, 0.0671, 0.0997,
          0.1056, 0.0346, 0.1231, 0.1626, 0.3652, 0.3262, 0.2995, 0.2109, 0.2104, 0.2085, 0.2282, 0.0747, 0.1969,
          0.4086, 0.6385, 0.7970, 0.7508, 0.5517, 0.2214, 0.4672, 0.4479, 0.2297, 0.3235, 0.4480, 0.5581, 0.6520,
          0.5354, 0.2478, 0.2268, 0.1788, 0.0898, 0.0536, 0.0374, 0.0990, 0.0956, 0.0317, 0.0142, 0.0076, 0.0223,
          0.0255, 0.0145, 0.0233, 0.0041, 0.0018, 0.0048, 0.0089, 0.0085],
         [0.0414,0.0436,0.0447,0.0844,0.0419,0.1215,0.2002,0.1516,0.0818,0.1975,0.2309,0.3025,0.3938,0.5050,0.5872,0.6610,0.7417,0.8006,0.8456,0.7939,0.8804,0.8384,0.7852,0.8479,0.7434,0.6433,0.5514,0.3519,0.3168,0.3346,0.2056,0.1032,0.3168,0.4040,0.4282,0.4538,0.3704,0.3741,0.3839,0.3494,0.4380,0.4265,0.2854,0.2808,0.2395,0.0369,0.0805,0.0541,0.0177,0.0065,0.0222,0.0045,0.0136,0.0113,0.0053,0.0165,0.0141,0.0077,0.0246,0.0198],
         ], dtype=float)
    output=list(pipeline.predict(new_samples))
    score = accuracy_score(original_labels, list(output))
    print('New samples accuracy: {0:f}'.format(score))
    #We get the label
    for a in output:
      print(a)

if __name__ == '__main__':
  tf.app.run()
