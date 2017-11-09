import numpy as np


def tsp_loader(fname):
    lines = []
    with open(fname, 'r') as f:
        for l in f:
            lines.append(l)
    lines = lines[5:-2]
    ret = np.zeros((len(lines), 2))
    for i, l in enumerate(lines):
        coords = l.split(' ')[1:]
        ret[i][0] = float(coords[0])
        ret[i][1] = float(coords[1][:-1])

    return ret

class DataSet:

    def __init__(self, file_name, delimiter, set_dist, random=True, normalize=False):
        self.features = 0
        self.classes = 0

        self.all = SubSet([], [])
        self.training = SubSet([], [])
        self.evaluation = SubSet([], [])
        self.test = SubSet([], [])

        self.__read_file(file_name, delimiter)
        if normalize:
            self.normalize_data()
        if random:
            self.__distribute_sets_stochastic(set_dist + [1-sum(set_dist)])
        else:
            self.__distribute_sets_ordered(set_dist)

    def get_mini_batches(self, batch_size):
        ret = []
        n_its = len(self.training)//batch_size
        for i in range(n_its):
            if i < n_its-1:
                ret.append(SubSet(self.training.x[i*batch_size:(i+1)*batch_size],
                                  self.training.y[i*batch_size:(i+1)*batch_size]))
            else:
                ret.append(SubSet(self.training.x[i*batch_size:], self.training.y[i*batch_size:]))
        return ret

    def __read_file(self, file_name, delimiter):
        size = 0
        out_max = 0
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                sline = line.split(delimiter)
                self.all.x.append(np.array([float(l)for l in sline[:-1]]))
                self.all.y.append(int(sline[-1]))
                size += 1
                if self.all.y[-1] > out_max:
                    out_max = self.all.y[-1]
        self.size = size
        self.classes = out_max+1
        self.features = len(self.all.x[0])

    def __distribute_sets_stochastic(self, set_dist):
        sets = [self.training, self.evaluation, self.test, SubSet([], [])]
        for x, y in self.all:
            s = np.random.choice(np.arange(0, 4, dtype=np.int8), p=set_dist)
            sets[s].x.append(x)
            sets[s].y.append(self.int_to_one_hot(y, self.classes))

    def __distribute_sets_ordered(self, set_dist):
        sets = [self.training, self.evaluation, self.test]
        d = []
        [d.append(int(sd*self.size+(d[-1] if len(d) > 0 else 0))) for sd in set_dist]
        s = 0
        for i, (x, y) in enumerate(self.all):
            if i > d[s]:
                s += 1
                if s == 3:
                    break
            sets[s].x.append(x)
            sets[s].y.append(self.int_to_one_hot(y, self.classes))

    def normalize_data(self):
        """
        f_max = np.max(self.all.x, 1)
        f_min = np.min(self.all.x, 1)
        for i, r in enumerate(self.all.x):
            tmp = np.zeros(self.features)
            for j, f in enumerate(r):
                tmp[j] = f-f_min[j]/(f_max[j]-f_min[j])
            self.all.x[i] = tmp
        """
        f_std = np.std(self.all.x, 1)
        f_mean = np.mean(self.all.x, 1)
        for i, r in enumerate(self.all.x):
            tmp = np.zeros(self.features)
            for j, f in enumerate(r):
                tmp[j] = f - f_mean[j] / f_std[i]
            self.all.x[i] = tmp


    @staticmethod
    def int_to_one_hot(value, size):
        ret = np.zeros(size)
        ret[value] = 1
        return ret


class SubSet:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return ((x, y) for x, y in zip(self.x, self.y))
