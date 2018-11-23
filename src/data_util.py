import os
import numpy as np
from mxnet import nd

number_str = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
data_dir = "data"

train_dir = os.path.join(data_dir, "train")
dev_dir = os.path.join(data_dir, "dev")
test_dir = os.path.join(data_dir, "test")
output_dir = "results"

def output_csv(y, filename):
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write("id,classes\n")
        for i in range(y.shape[0]):
            f.write(str(i+1)+','+str(int(y[i].asscalar()))+'\n')


class Data():
    '''
    content: [(label, f0, engy), ...]
    '''
    def __init__(self, file_dir, labeled=True):
        def get_content(dir):
            fs = sorted(list(set([i.split('.')[0] for i in os.listdir(dir)])), key=lambda x: int(x) if x.isdigit() else x)
            ret = []
            for f in fs:
                with open(os.path.join(dir, f+".f0")) as fp:
                    f0 = [eval(line) for line in fp.readlines()]
                with open(os.path.join(dir, f+".engy")) as fp:
                    engy = [eval(line) for line in fp.readlines()]
                ret.append((f, f0, engy))
            return ret

        self.labeled = labeled
        if labeled:
            self.content = []
            for tone in range(1, 5):
                cur_dir = os.path.join(file_dir, number_str[tone])
                self.content += get_content(cur_dir)
        else:
            self.content = get_content(file_dir)

    def getX(self, length):
        def pad(x, length):
            if len(x) > length:
                return x[:length]
            else:
                return x + [0] * (length - len(x))
        x = []
        for _, f0, engy in self.content:
            x.append(pad(f0, length))
            x.append(pad(engy, length))
        return nd.array(x).reshape((len(x) // 2, 2, -1))
        
    def getY(self):
        if self.labeled:
            return nd.array([int(label[-1]) - 1 for label, _, _ in self.content])
        else:
            raise "Cannot getY for unlabeled data!"

    def getfX(self, length):
        x = self.getX(length)
        # x[:, :, 1:, :] -= x[:, :, :-1, :]
        N, C, H = x.shape
        xx = x[:, 0, :]
        #xx -= nd.mean(xx, (0, 2), keepdims=True)
        #xx /= nd.mean(xx**2, (0, 2), keepdims=True)
        return xx.reshape((N, 1, H, 1))

    def max_f0_length(self):
        content = self.content
        return max(len(i[1]) for i in content)
    
    def max_engy_length(self):
        content = self.content
        return max(len(i[2]) for i in content)

    def info(self):
        return "len(f0) <= {}, len(engy) <= {}".format(self.max_f0_length(), self.max_engy_length())

data_train = Data(train_dir, labeled=True)
data_dev = Data(dev_dir, labeled=True)
data_test = Data(test_dir, labeled=False)

max_l = max([data_train.max_f0_length(), data_train.max_engy_length(),
            data_dev.max_f0_length(), data_dev.max_engy_length(),
            data_test.max_f0_length(), data_test.max_engy_length()])

if __name__ == "__main__":
    print("data_train_info: ", data_train.info())
    print("data_dev_info: ", data_dev.info())
    print("data_test_info: ", data_test.info())
