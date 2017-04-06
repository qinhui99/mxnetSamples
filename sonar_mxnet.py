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

# net = mx.sym.Variable('data')
# net = mx.sym.FullyConnected(net, name='fc1', num_hidden=120)
# net = mx.sym.Activation(net, name='relu1', act_type="relu")
# net = mx.sym.FullyConnected(net, name='fc2', num_hidden=2)
# net = mx.sym.SoftmaxOutput(net, name='softmax',multi_output=True,)

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=70)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=10)
net = mx.sym.Activation(net, name='relu2', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc3', num_hidden=2)
net = mx.sym.SoftmaxOutput(net, name='softmax',multi_output=False,)

batch_size = 10

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

# We use utils function in sklearn to get iris dataset in pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
qhdatas,qh_target = loadDatas()

# Normalize data
scalar=StandardScaler()
scalared_data=scalar.fit_transform(qhdatas)
# shuffle data
X, y = shuffle(scalared_data, qh_target)

# split data
split_index=160
# train_data = X[:split_index, :].astype('float32')
#
# train_label = y[:split_index]
# val_data = X[split_index :].astype('float32')
# val_label = y[split_index:]

split_index2=170
batch_size = split_index2-split_index
#
train_data =np.concatenate((X[:split_index],X[split_index2:]),axis=0).astype('float32')
train_label = np.append(y[:split_index],y[split_index2:])
print ("train_data.shape",train_data.shape)
print (train_label.shape)
val_data = X[split_index :split_index2].astype('float32')
val_label = y[split_index:split_index2]

# Build iterator
train_iter = mx.io.NDArrayIter(data=train_data, label=train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data=val_data, label=val_label, batch_size=batch_size, shuffle=True)
logging.basicConfig(level=logging.INFO)
epoch_num=20
lr=0.07
devs = mx.cpu()
mod = mx.model.FeedForward(
    ctx=devs,
    symbol=net,
    num_epoch=epoch_num,
    learning_rate=lr,
    optimizer='sgd')

mod.fit(X = train_iter, eval_metric = 'acc',  eval_data=val_iter)
print('Finished training')

def CalAcc(pred_prob, labels):
    # print ("pred_prob",pred_prob)
    pred = np.argmax(pred_prob, axis=1)
    # print ("predition:",pred)
    # print ("label:",labels)
    return np.sum(pred == labels) * 1.0/len(labels)


print (mod.score(val_iter, ['ce', 'acc'],num_batch=None))
# mod.save("qh",epoch=epoch_num)

#labels:
original_labels = np.array([0,1,0,0])
new_samples = np.array(
      [[0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032],
       [0.0629,0.1065,0.1526,0.1229,0.1437,0.1190,0.0884,0.0907,0.2107,0.3597,0.5466,0.5205,0.5127,0.5395,0.6558,0.8705,0.9786,0.9335,0.7917,0.7383,0.6908,0.3850,0.0671,0.0502,0.2717,0.2839,0.2234,0.1911,0.0408,0.2531,0.1979,0.1891,0.2433,0.1956,0.2667,0.1340,0.1073,0.2023,0.1794,0.0227,0.1313,0.1775,0.1549,0.1626,0.0708,0.0129,0.0795,0.0762,0.0117,0.0061,0.0257,0.0089,0.0262,0.0108,0.0138,0.0187,0.0230,0.0057,0.0113,0.0131],
       [0.0181,0.0146,0.0026,0.0141,0.0421,0.0473,0.0361,0.0741,0.1398,0.1045,0.0904,0.0671,0.0997,0.1056,0.0346,0.1231,0.1626,0.3652,0.3262,0.2995,0.2109,0.2104,0.2085,0.2282,0.0747,0.1969,0.4086,0.6385,0.7970,0.7508,0.5517,0.2214,0.4672,0.4479,0.2297,0.3235,0.4480,0.5581,0.6520,0.5354,0.2478,0.2268,0.1788,0.0898,0.0536,0.0374,0.0990,0.0956,0.0317,0.0142,0.0076,0.0223,0.0255,0.0145,0.0233,0.0041,0.0018,0.0048,0.0089,0.0085],
       [0.0235,0.0291,0.0749,0.0519,0.0227,0.0834,0.0677,0.2002,0.2876,0.3674,0.2974,0.0837,0.1912,0.5040,0.6352,0.6804,0.7505,0.6595,0.4509,0.2964,0.4019,0.6794,0.8297,1.0000,0.8240,0.7115,0.7726,0.6124,0.4936,0.5648,0.4906,0.1820,0.1811,0.1107,0.4603,0.6650,0.6423,0.2166,0.1951,0.4947,0.4925,0.4041,0.2402,0.1392,0.1779,0.1946,0.1723,0.1522,0.0929,0.0179,0.0242,0.0083,0.0037,0.0095,0.0105,0.0030,0.0132,0.0068,0.0108,0.0090],
       ], dtype=float)

result,datas,labels=mod.predict(mx.io.NDArrayIter(data=new_samples,label=original_labels,batch_size=1),return_data=True)
print ("New samples accuracy:",CalAcc(result,original_labels))
#we get the labels:
for a in result :
    max=np.argmax(a)
    print (max)

for x in labels :
    print (x)