import numpy as np
from data_reader import *
from matplotlib import pyplot as plt
import time


class SOM:

    CHESS_BOARD_DISTANCE = lambda x: np.max(x)
    MANHATTAN_DISTANCE = lambda x: np.sum(x)
    SQUARED_EUCLIDEAN_DISTANCE = lambda x: np.sum(np.square(x))
    EUCLIDEAN_DISTANCE = lambda x: np.sqrt(SOM.SQUARED_EUCLIDEAN_DISTANCE(x))

    def __init__(self, n_features, network_shape, distance_fn, wraparound):
        self.n_features = n_features
        self.shape = network_shape
        #self.radius = np.max(network_shape)/2

        self.weights = np.random.rand(np.prod(network_shape), n_features)

        dm = self._generate_distance_matrix(network_shape,wraparound, distance_fn)

        self.distance_matrix = -(dm**2)
        self.disatance_mean = np.mean(dm, 1)

        self.t = 0
        self.lr0 = 0.1

        self.lambda_n = 2**12
        self.lambda_lr = self.lambda_n**2

        self.sigma0 = np.sum(self.shape)/2


    def train_step(self, input_data):
        input_weight_distance_matrix = self._input_weight_distance(input_data)
        discriminant_vector = self._generate_discriminant(input_weight_distance_matrix)
        winner = np.argmin(discriminant_vector)
        tn = self.get_tn(winner)

        delta_w = np.zeros(self.weights.shape)
        for i in range(len(self.weights)):
            delta_w[i] = self.learning_rate() * tn[i] * input_weight_distance_matrix[i]
        self.weights += delta_w

        self.t += 1

        return delta_w

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

    def total_distance(self):
        total_distance = 0
        for i in range(self.weights.shape[0]-1):
            total_distance += np.sqrt(np.sum(np.square(self.weights[i-1] - self.weights[i])))
        return total_distance

    @staticmethod
    def _generate_distance_matrix(shape, wrap=True, distance_fn=MANHATTAN_DISTANCE):
        n_nodes = np.prod(shape)
        ret = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            md_i = np.unravel_index(i, shape)
            for j in range(n_nodes):
                distances = np.abs(np.subtract(md_i, np.unravel_index(j, shape)))
                # neighbourhood matrix wraparound
                if wrap:
                    for k, d in enumerate(shape):
                        if distances[k] > shape[k]/2:
                            distances[k] = shape[k] - distances[k]
                distance = distance_fn(distances)
                ret[i][j] = (distance)
        return ret


def tsp_man(case, node_factor, lr, lambda_n, lambda_lr, sigma, visualize=True):
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

        x /= 2
        y = np.sqrt((dim_radius[0]/2)**2-x**2)
        x += dim_center[0]

        if i <= nn/2:
            y += dim_center[1]
            net.weights[i] = [x, y]
        else:
            y *= -1
            y += dim_center[1]
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
        deltas = []
        for d in range(data.shape[0]):
            deltas.append(net.train_step(data[np.random.randint(0, data.shape[0])]))


        i += 1
        if i % 25 == 0:
            total_distance = net.total_distance()
            if np.abs(total_distance-total_distance_prev) < 0.5:
                return total_distance
            total_distance_prev = total_distance
            if visualize:
                path.set_data(np.concatenate((net.weights[:,0], [net.weights[0, 0]])),
                              np.concatenate((net.weights[:, 1], [net.weights[0, 1]])))
                plt.pause(0.005)
