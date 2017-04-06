#encoding=utf-8
"""
[Author] hui qin
[E-mail] qinhui99@hotmail.com
[Create] 2017/04
[Last Modified] 2017/04/04
"""
import mxnet as mx

import logging
import numpy as np
from sklearn import cross_validation
# mlp

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=12)
net = mx.sym.Activation(net, name='relu1', act_type="relu")

net = mx.sym.FullyConnected(net, name='fc2', num_hidden=3)
net = mx.sym.SoftmaxOutput(net, name='softmax',multi_output=False,)


batch_size = 10

# We use utils function in sklearn to get iris dataset in pickle
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

# Normalize data
scalar=StandardScaler()
scalared_data=scalar.fit_transform(iris.data)
# shuffle data
X, y = shuffle(scalared_data, iris.target)
# X, y = (scalared_data, iris.target)
# split dataset
train_data = X[:120, :].astype('float32')

train_label = y[:120]
val_data = X[120 :].astype('float32')
val_label = y[120:]

# Build iterator
train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data=val_data, label=val_label, batch_size=batch_size, shuffle=True)

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

logging.basicConfig(level=logging.INFO)

epoch_num=20
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='adam',
        optimizer_params={'learning_rate':.01},
        eval_metric='acc',
        num_epoch=epoch_num)

print('Finished training')

def CalAcc(pred_prob, labels):
    # print ("pred_prob",pred_prob)
    pred = np.argmax(pred_prob, axis=1)
    # print ("predition:",pred)
    # print ("label:",labels)
    return np.sum(pred == labels) * 1.0/len(labels)

#The labels should be :[1,0,2,1,0,1,1,1,2,1]
original_labels = np.array([1,0,2,1,0,1,1,1,2,1])
new_samples = np.array(
    [[5.5, 2.3, 4.0, 1.3], [5.0, 3.3, 1.4, 0.2], [5.8, 2.7, 5.1, 1.9], [5.5, 2.6, 4.4, 1.2],[4.9,3.1,1.5,0.1],
     [4.9,2.4,3.3,1.0], [6.0,2.2,4.0,1.0], [6.6,2.9,4.6,1.3], [6.4, 2.8, 5.6, 2.1], [5.5,2.3,4.0,1.3]],
    dtype=float)


result=mod.predict(mx.io.NDArrayIter(data=new_samples,label=None,batch_size=batch_size, shuffle=False), num_batch=1, reset=False)
print ("New samples accuracy:",CalAcc(result.asnumpy(),original_labels))

#the predicted results:
for a in result.asnumpy():
    max = np.argmax(a)
    print (max),
print

# print (mod.score(val_iter, ['ce', 'acc'],num_batch=None))
# mod.symbol.save("iris-symbol.json")
# mod.save_params("iris-00"+str(epoch_num)+".params")