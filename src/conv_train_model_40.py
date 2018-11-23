from data_util import *
from config import *
import mxnet as mx
from mxnet import autograd as ag, nd, gluon, init
from mxnet.gluon import nn, trainer
from time import time
# prepare data

batch_size = 30
train_data = mx.io.NDArrayIter(data_train.getdX(max_l), data_train.getY(), batch_size, shuffle=True)
dev_data_X = data_dev.getdX(max_l)
dev_data_Y = data_dev.getY()
test_data = data_test.getdX(max_l)

# build the net
net = nn.HybridSequential()

conv2d_0_config = {
    "channels": 64,
    "kernel_size": (2, 2),
    "padding": (0, 0),
    "activation": "relu",
    "use_bias": True
}

conv2d_1_config = {
    "channels": 64,
    "kernel_size": (2, 1),
    "padding": (0, 0),
    "activation": "relu",
    "use_bias": True
}

conv2d_2_config = {
    "channels": 32,
    "kernel_size": (2, 1),
    "padding": (0, 0),
    "activation": "relu",
    "use_bias": True
}

conv2d_3_config = {
    "channels": 16,
    "kernel_size": (2, 1),
    "padding": (0, 0),
    "activation": "relu",
    "use_bias": True
}

conv2d_4_config = {
    "channels": 8,
    "kernel_size": (2, 1),
    "padding": (0, 0),
    "activation": "relu",
    "use_bias": True
}

conv2d_5_config = {
    "channels": 8,
    "kernel_size": (2, 1),
    "padding": (0, 0),
    "activation": "relu",
    "use_bias": True
}

dor = 0.0

with net.name_scope():
    net.add(nn.Conv2D(**conv2d_0_config))
    net.add(nn.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
    net.add(nn.Dropout(dor))
    net.add(nn.Conv2D(**conv2d_1_config))
    net.add(nn.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))    
    net.add(nn.Dropout(dor))
    net.add(nn.Conv2D(**conv2d_2_config))
    net.add(nn.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
    net.add(nn.Dropout(dor))    
    net.add(nn.Conv2D(**conv2d_3_config))
    net.add(nn.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
    net.add(nn.Dropout(dor))
    net.add(nn.Conv2D(**conv2d_4_config))
    net.add(nn.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
    net.add(nn.Dropout(dor))
    net.add(nn.Conv2D(**conv2d_5_config))
    net.add(nn.MaxPool2D(pool_size=(2, 1), strides=(2, 1)))
    net.add(nn.Dropout(dor))
    net.add(nn.Dense(4))

net.hybridize()

# train it
gpus = mx.test_utils.list_gpus()
ctx =  [mx.gpu()] if gpus else [mx.cpu(0)]
net.initialize(mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4})

epoch = 1000
# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()
softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
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
    if acc > 0.99:
        break
    # Calc dev-acc
y_test = net(test_data)
pred = nd.exp(y_test)/nd.sum(nd.exp(y_test), 1, keepdims=True)
print(nd.argmax(pred, 1) + 1)
"""print(net.collect_params()['hybridsequential0_conv0_weight'].data())
print(net.collect_params()['hybridsequential0_conv0_bias'].data())
print(net.collect_params()['hybridsequential0_conv1_weight'].data())
print(net.collect_params()['hybridsequential0_conv1_bias'].data())
"""
output_csv(nd.argmax(pred, 1) + 1, "try.csv")