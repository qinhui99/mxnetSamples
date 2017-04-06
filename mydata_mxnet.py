#encoding=utf-8
"""
[Author] hui qin
[E-mail] qinhui99@hotmail.com
[Create] 2017/04
[Last Modified] 2017/04/04
"""

import mxnet as mx
import csv
import logging
import numpy as np
# mlp
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=4)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=3)
net = mx.sym.SoftmaxOutput(net, name='softmax')

batch_size = 10

def CalAcc(pred_prob, labels):
    # print ("pred_prob",pred_prob)
    pred = np.argmax(pred_prob, axis=1)
    # print ("predition:",pred)
    # print ("label:",labels)
    return np.sum(pred == labels) * 1.0/len(labels)

#10000,4,d30,d45,d60
def loadDatas():
    with open("datas.txt","r") as csv_file:
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

# We use utils function in sklearn to get iris dataset in pickle

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
qhdatas,qh_target = loadDatas()

# Normalize data
scalar=StandardScaler()
scalared_data=scalar.fit_transform(qhdatas)
# shuffle data
X, y = shuffle(scalared_data, qh_target)
# X, y = (scalared_data, iris.target)
# split data
split_index=8000
train_data = X[:split_index, :].astype('int')

train_label = y[:split_index]
val_data = X[split_index :].astype('int')
val_label = y[split_index:]

# Build iterator
train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data=val_data, label=val_label, batch_size=batch_size, shuffle=True)
logging.basicConfig(level=logging.INFO)
epoch_num=15
lr=0.1
devs = mx.cpu()
mod = mx.mod.Module(symbol=net,
                    context=devs,
                    data_names=['data'],
                    label_names=['softmax_label'])
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='adam',
        optimizer_params={'learning_rate':lr},
        eval_metric='acc',
        num_epoch=epoch_num)
print('Finished training')


print (mod.score(val_iter, ['ce', 'acc'],num_batch=None))
# mod.save("qh",epoch=epoch_num)

# the label is :[0,0,1,2,0,2,1,1,2,0]
original_labels = np.array([0, 0, 1, 2, 0, 2, 1, 1, 2, 0])
new_samples = np.array(
      [[14,5,30], [24,5,30], [4,15,45], [4,15,60], [23,5,30],
        [14,5,60], [24,5,45], [4,15,45], [4,15,60], [23,5,30]
       ],
    dtype=int)


result=mod.predict(mx.io.NDArrayIter(data=new_samples,label=None,batch_size=batch_size, shuffle=False),
                   num_batch=1, reset=False,)
print ("New samples accuracy:",CalAcc(result.asnumpy(),original_labels))

for a in result.asnumpy() :
    max=np.argmax(a)
    print (max),