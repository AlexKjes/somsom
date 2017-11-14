import numpy as np
from data_reader import *
from matplotlib import pyplot as plt
from matplotlib import colors as clr
import time
import test
import utility


class SOM:

    def __init__(self, n_features, network_shape, distance_fn, wraparound):
        self.n_features = n_features
        self.shape = network_shape
        #self.radius = np.max(network_shape)/2

        self.weights = np.random.rand(np.prod(network_shape), n_features)

        dm = utility.generate_distance_matrix(network_shape, wraparound, distance_fn)

        self.distance_matrix = -(dm**2)
        self.disatance_mean = np.mean(dm, 1)

        self.t = 0
        self.lr0 = 0.1

        self.lambda_n = 2**12
        self.lambda_lr = self.lambda_n**2

        self.sigma0 = np.sum(self.shape)/2

        self.prev_delta_sum = 0

        self.trained = False

    def train_step(self, data_set):
        np.random.shuffle(data_set)
        for d in data_set:
            input_weight_distance_matrix = self._input_weight_distance(d)
            discriminant_vector = self._generate_discriminant(input_weight_distance_matrix)
            winner = np.argmin(discriminant_vector)
            tn = self.get_tn(winner)

            delta_w = np.zeros(self.weights.shape)
            for i in range(len(self.weights)):
                delta_w[i] = self.learning_rate() * tn[i] * input_weight_distance_matrix[i]

            self.weights += delta_w

        if self.t == max(self.lambda_lr, self.lambda_n):
            self.trained = True

        self.t += 1


    def learning_rate(self):
        return self.lr0 * np.exp((-self.t)/self.lambda_lr)

    def get_tn(self, winner_index):
        #distance = np.sum(np.abs((self.weights - self.weights[winner_index])),1)
        """self.distance_matrix[winner_index]"""
        sigma = self.sigma0*np.exp(-self.t/self.lambda_n)
        return np.exp(self.distance_matrix[winner_index]/(sigma))


    def _generate_discriminant(self, distance_matrix):
        return np.sum(np.abs(distance_matrix), 1)

    def _input_weight_distance(self, x):
        #ret = np.zeros(self.weights.shape)
        #for i in range(self.weights.shape[0]):
        #    np.subtract(x, self.weights[i], ret[i])
        #return ret
        return x - self.weights


def map_som_to_data(data, weights):
    wd_map = np.zeros(data.shape[0])
    for i, d in enumerate(data):
        wd_map[i] = np.argmin(np.sum(np.abs(d-weights), axis=1))

    wd_sec = np.zeros(wd_map.shape, dtype=np.int8)
    for i, w in enumerate(wd_map):
        m = np.argmax(wd_map)
        wd_sec[m] = wd_map.shape[0]-i-1
        wd_map[m] = -1
    return wd_sec


def total_distance(data, weights):
    """
    path = map_som_to_data(data, weights)
    td = 0
    for i, c in enumerate(path[:-1]):
        td += np.sqrt(np.sum(np.square(data[path[i]]-data[path[i+1]])))
    td += np.sqrt(np.sum(np.square(data[path[-1]]-data[path[0]])))

    print(td)
    return td
    """
    td = 0
    for i in range(weights.shape[0]-1):
        td += np.sqrt(np.sum(np.square(weights[i] - weights[i+1])))
    td += np.sqrt(np.sum(np.square(weights[-1] - weights[0])))
    return td

def tsp_man(case, node_factor, lr, lambda_n, lambda_lr, sigma, radiusfactor=1, translate_x=1, translate_y=1, visualize=True, k=25):

    data = tsp_loader('data/tspe/{}.txt'.format(case))
    net = SOM(2, [int(data.shape[0]*node_factor)], SOM.MANHATTAN_DISTANCE, True)

    ## set params
    net.lr0 = lr
    net.sigma0 = sigma
    net.lambda_n = lambda_n
    net.lambda_lr = lambda_lr


    nn = net.weights.shape[0]
    dim_max = np.max(data, 0)
    i_max = np.argmax(dim_max)
    dim_min = np.min(data, 0)
    dim_center = (dim_max + dim_min) / 2
    dim_diam = (dim_max - dim_min)
    dim_radius = dim_diam / 2
    scale = (dim_max-dim_min)/(nn/2)

    for i, x in enumerate(np.concatenate((np.arange(-dim_radius[i_max], dim_radius[i_max], scale[i_max]),
                                         np.arange(dim_radius[i_max], -dim_radius[i_max], -scale[i_max])))[:nn]):

        x *= radiusfactor
        y = np.sqrt((dim_radius[0]*radiusfactor)**2-x**2)
        x += dim_center[0] * translate_x

        if i <= nn/2:
            y += dim_center[1] * translate_y
            net.weights[i] = [x, y]
        else:
            y *= -1
            y += dim_center[1] * translate_y
            net.weights[i] = [x, y]



    if visualize:
        plt.scatter(data[:, 0], data[:, 1], marker='*', facecolor='r')
        path = plt.plot(np.concatenate((net.weights[:,0], [net.weights[0, 0]])),
                              np.concatenate((net.weights[:, 1], [net.weights[0, 1]])))[0]
        #plt.xlim(dim_min[0], dim_max[0])
        #plt.ylim(dim_min[1], dim_max[1])
        plt.ion()
        plt.show()
    i = 0
    total_distance_prev = 0
    while True:
        net.train_step(data)


        i += 1
        if i % k == 0:
            td = total_distance(data, net.weights)
            if np.abs(td-total_distance_prev) < 0.5:
                return td
            total_distance_prev = td
            if visualize:
                print(td)
                path.set_data(np.concatenate((net.weights[:,0], [net.weights[0, 0]])),
                              np.concatenate((net.weights[:, 1], [net.weights[0, 1]])))
                plt.pause(0.005)


def gen_distance_matrix(net):

    ret = np.zeros(net.shape)
    wm = net.weights.reshape((net.shape[0], net.shape[1], net.n_features))

    for y, r in enumerate(wm):
        for x, _ in enumerate(r):

            n = []
            if x > 0 < y: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y - 1, x - 1]))))
            if 0 < y: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y - 1, x]))))
            if 0 < y and x < net.shape[1] - 1: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y - 1, x + 1]))))
            if x > 0: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y, x - 1]))))
            if x < net.shape[1] - 1: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y, x + 1]))))
            if x > 0 and y < net.shape[0] - 1: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y + 1, x - 1]))))
            if y < net.shape[0] - 1: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y + 1, x]))))
            if y < net.shape[0] - 1 and x < net.shape[1] - 1: n.append(np.sqrt(np.sum(np.square(wm[y, x] - wm[y + 1, x + 1]))))
            ret[y, x] = np.mean(n)

    mi = np.min(ret)
    ret = -((ret-mi)/(np.max(ret)-mi))+1

    return ret


def map_minst(net, data, features, colors):

    cm = class_map(net, data, features)

    return [[clr.to_rgb(colors[v]) for v in r] for r in cm]


def class_map(net, data, features):

    n_unique_features = np.unique(features).shape[0]
    blabla = np.zeros((net.shape[0], net.shape[0], n_unique_features))

    for d, f in zip(data, features):
        bmu_index = np.unravel_index(np.argmin(np.sum(np.abs(d-net.weights), 1)), net.shape)
        blabla[bmu_index][f] += 1

    ret = np.zeros((net.shape[0], net.shape[0]), np.int64)
    for y, r in enumerate(blabla):
        for x, v in enumerate(r):
            ret[y, x] = np.argmax(blabla[y, x])

    return ret

def evaluate_som(net, data, features, test_data, test_features):

    cm = class_map(net, data, features)
    correct = 0
    for td, tf in zip(test_data, test_features):
        bmu_index = np.unravel_index(np.argmin(np.sum(np.abs(td - net.weights), 1)), net.shape)
        if cm[bmu_index] == tf:
            correct += 1

    return correct/test_data.shape[0]



def img_man(data, output_shape, visualize=True, k=1, lr0=.2, lambda_n=10, lambda_lr=100, sigma=10,
            features=None, test_data=None, test_features=None):

    filter = np.sum(data.T, 1).T
    #filter[filter>0] = 1

    colors = ['red', 'orange', 'lime', 'blue', 'gray', 'yellow', 'brown', 'green', 'pink', 'magenta']

    net = SOM(data.shape[1], output_shape, utility.euclidean_distance, False)

    #net.weights *= filter


    net.lr0 = lr0
    net.lambda_n = lambda_n
    net.lambda_lr = lambda_lr
    net.sigma0 = sigma


    if visualize:

        img = plt.imshow(gen_distance_matrix(net))
        plt.clim(0, 1)
        plt.colorbar()

        plt.ion()
        plt.show()


    i=0

    while not net.trained:
        print('epoch: {}'.format(i))
        print('learning rate: {}'.format(net.learning_rate()))
        #print(net.get_tn(net.weights.shape[0]//2).shape)
        #print(net.get_tn(net.weights.shape[0]//2).reshape(net.shape))
        start_time = time.time()
        net.train_step(data)
        #print("iteration time: {}".format(time.time()-start_time))

        i += 1
        if i % k == 0:

            if visualize:
                start_time = time.time()
                img.set_data(gen_distance_matrix(net))
                plt.pause(0.005)
                #print("visualization time: {}".format(time.time() - start_time))


    if visualize and features is not None:
        mm = map_minst(net, data, features, colors)
        plt.figure(2)
        #img.set_data(mm)
        plt.imshow(mm)
        print("error rate: {}".format(evaluate_som(net, data,features, test_data, test_features)))


        plt.pause(0.005)

