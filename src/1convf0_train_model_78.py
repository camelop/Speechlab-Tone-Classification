from data_util import *
from config import *
import mxnet as mx
from mxnet import autograd as ag, nd, gluon, init
from mxnet.gluon import nn, trainer
from time import time
# prepare data

batch_size = 40
train_data = mx.io.NDArrayIter(data_train.getfX(max_l), data_train.getY(), batch_size, shuffle=True)
dev_data_X = data_dev.getfX(max_l)
dev_data_Y = data_dev.getY()
test_data = data_test.getfX(max_l)

# build the net
net = nn.HybridSequential()

feature_dense = 1
epoch = 1000
window_size = 128
channels = 128 # 128, 78.9%
dropout_rate = 0.0
learning_rate = 1e-3
stop_after = 1000

conv2d_0_config = {
    "channels": channels,
    "kernel_size": (window_size, 1),
    "padding": (window_size, 0),
    "activation": "relu",
    "use_bias": True
} 


with net.name_scope():
    net.add(nn.BatchNorm())
    net.add(nn.Dense(feature_dense, flatten=False))
    net.add(nn.BatchNorm())
    net.add(nn.Conv2D(**conv2d_0_config))
    net.add(nn.MaxPool2D(pool_size=(max_l+1+window_size, 1), strides=(window_size, 1)))
    net.add(nn.BatchNorm())
    net.add(nn.Dense(4))

net.hybridize()

# train it
gpus = mx.test_utils.list_gpus()
ctx =  [mx.gpu()] if gpus else [mx.cpu(0)]
net.initialize(mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

last_dev = -1
cnt = 0

for i in range(epoch):
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # Backpropagate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
    # Gets the evaluation result.
    name, acc = metric.get()
    # Reset evaluation result to initial state.
    metric.reset()
    dev_acc = nd.mean(nd.argmax(net(dev_data_X), 1) == dev_data_Y)
    print('training acc at epoch %d: %s=%f, dev_acc:%f'%(i, name, acc, dev_acc.asscalar()))
    if dev_acc > 0.99:
        break
    if last_dev >= dev_acc.asscalar():
        cnt += 1
        if cnt > stop_after:
            break
    else:
        last_dev = dev_acc.asscalar()
        cnt = 0
    # Calc dev-acc
y_test = net(test_data)
pred = nd.exp(y_test)/nd.sum(nd.exp(y_test), 1, keepdims=True)
print(net.collect_params())
"""print(net.collect_params()['hybridsequential0_conv0_weight'].data())
print(net.collect_params()['hybridsequential0_conv0_bias'].data())
print(net.collect_params()['hybridsequential0_conv1_weight'].data())
print(net.collect_params()['hybridsequential0_conv1_bias'].data())
"""
ans = nd.argmax(pred, 1) + 1
output_csv(ans, "try.csv")

print(ans)
print(1, nd.mean(ans==1).asscalar())
print(2, nd.mean(ans==2).asscalar())
print(3, nd.mean(ans==3).asscalar())
print(4, nd.mean(ans==4).asscalar())