import tensorflow as tf
import numpy as np
import utility


from data_reader import *
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patches

class TfSelfOrganizingMap:

    def __init__(self, input_size, output_shape, wraparound=False, distance_fn=utility.euclidean_distance,
                 lr0=0.1, sigma=None, lambda_n=None, lambda_lr=None, n_iterations=1000, model_path='model/som_model2',
                 load=False):

        # useful stuff
        self.input_size = input_size
        self.output_size = output_shape
        self.neighbour_distances = utility.generate_distance_matrix(output_shape, wraparound, distance_fn)
        self.n_output_nodes = np.prod(output_shape)
        self.isTrained = False
        self.model_path = model_path

        # define som parameters
        self.lr0 = lr0
        self.sigma = sigma if sigma is not None else np.max(output_shape)/2
        self.lambda_n = lambda_n if lambda_n is not None else n_iterations/np.log(self.sigma)
        self.lambda_lr = lambda_lr if lambda_lr is not None else n_iterations
        self.n_iterations = n_iterations
        self.t = 0

        # tf definitions
        self.session = None
        # placeholders
        self.tf_input = tf.placeholder(tf.float64, [None, input_size])
        # variables
        self.tf_weights = tf.Variable(np.random.rand(self.n_output_nodes, input_size), dtype=tf.float64)
        self.tf_t = tf.Variable(0, dtype=tf.float64)
        # constants
        self.tf_neg_squared_neighbourhood_distance = tf.constant(-np.square(self.neighbour_distances), dtype=tf.float64)
        self.tf_lambda_n = tf.constant(self.lambda_n, dtype=tf.float64)
        self.tf_lambda_lr = tf.constant(self.lambda_lr, dtype=tf.float64)
        self.tf_sigma = tf.constant(self.sigma, dtype=tf.float64)
        # ops
        self.tf_difference_op = tf.map_fn(lambda x: x - self.tf_weights, self.tf_input)
        self.tf_bmu = tf.argmin(tf.reduce_sum(tf.abs(self.tf_difference_op), 2), 1)
        self.tf_lr_t_op = lr0 * tf.exp(tf.divide(tf.negative(self.tf_t), self.tf_lambda_lr))
        self.tf_neighbourhood_t_op = self.tf_sigma * tf.exp(tf.divide(tf.negative(self.tf_t), self.tf_lambda_n))
        self.tf_neighbourhood_factor_op = tf.exp(tf.gather(self.tf_neg_squared_neighbourhood_distance, self.tf_bmu)/self.tf_neighbourhood_t_op)
        self.tf_update_factor = self.tf_lr_t_op * self.tf_neighbourhood_factor_op
        self.tf_factorize_diff_op = tf.expand_dims(self.tf_update_factor, 2) * self.tf_difference_op
        self.tf_update_weights_op = self.tf_weights.assign_add(tf.reduce_mean(self.tf_factorize_diff_op, 0))
        self.tf_increment_time_op = self.tf_t.assign_add(1)
        # saver
        self.tf_saver = tf.train.Saver([self.tf_t, self.tf_weights])

        if load:
            self.tf_saver.restore(self._get_sess(), model_path)
            self.t = self._get_sess().run(self.tf_t)

    def train(self, data, batch_size=1):
        ints = int(np.ceil(data.shape[0]/batch_size))
        np.random.shuffle(data)
        for j in range(ints):
            self._get_sess().run(self.tf_update_weights_op, feed_dict={self.tf_input: data[batch_size*j:batch_size*(j+1)]})
        self._get_sess().run(self.tf_increment_time_op)
        self.t += 1
        if self.t > self.n_iterations:
            self.isTrained = True

        if self.t % 1000 == 0:
            self.save_model()

    def set_initial_weights(self, weights):
        self._get_sess().run(self.tf_weights.assign(weights))

    def _get_sess(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        return self.session

    def get_lattice(self):
        ret = self._get_sess().run(self.tf_weights)
        return ret.reshape(np.concatenate((self.output_size, [self.input_size])))

    def save_model(self):
        print('Saving model..')
        self.tf_saver.save(self._get_sess(), self.model_path)


def tsp_man(tsp_problem):
    data = tsp_loader('data/tspe/{}.txt'.format(tsp_problem))

    net = TfSelfOrganizingMap(2, [data.shape[0]*3], wraparound=True, lr0=.9, n_iterations=1000)
    net.set_initial_weights(utility.weight_circle(data, net.n_output_nodes, radius_factor=.1))
    plt.scatter(data[:, 0], data[:, 1])
    weights = net.get_lattice()
    path, = plt.plot(weights[:, 0], weights[:, 1])
    plt.ion()
    plt.show()
    i = 0
    while not net.isTrained:

        net.train(data, 1)

        if i % 20 == 0:
            weights = net.get_lattice()
            path.set_data(weights[:, 0], weights[:, 1])
            plt.pause(0.05)
            #print(net.t * data.shape[0])
            print(utility.total_distance(net.get_lattice()))
        i += 1

    for _ in range(1000):

        net.train(data, 1)

        if i % 20 == 0:
            weights = net.get_lattice()
            path.set_data(weights[:, 0], weights[:, 1])
            plt.pause(0.05)
            print(utility.total_distance(net.get_lattice()))

        i += 1

    input()


def img_man(data, lattice, features, n_iterations, lr, load):

    data_copy = np.copy(data)

    net = TfSelfOrganizingMap(data.shape[1], lattice, n_iterations=n_iterations, lr0=lr, load=load)


    contour_img = plt.imshow(utility.generate_contour_map(net.get_lattice()))
    plt.colorbar()
    plt.ion()
    plt.figure(2)
    [plt.plot(-1, -1, c=c, label=str(i), marker='s', linestyle='') for i, c in enumerate(utility.colors)]
    #plt.xlim(0, lattice[0]+0.5)
    #plt.ylim(0, lattice[1]+0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    test_img = plt.imshow(utility.map_mnist(net.get_lattice(), data, features))

    plt.show()

    i = 0


    while not net.isTrained:

        net.train(data_copy)

        if i % 20 == 0:
            contour_img.set_data(utility.generate_contour_map(net.get_lattice()))
            tmp = utility.map_mnist(net.get_lattice(), data, features)
            print(tmp)
            test_img.set_data(tmp)
            plt.pause(0.5)
            print(net.t)
            print(utility.evaluate_som2(net.get_lattice(), data, features))

        i += 1



    for _ in range(10000):

        net.train(data_copy)

        if i % 20 == 0:
            contour_img.set_data(utility.generate_contour_map(net.get_lattice()))
            test_img.set_data(utility.map_mnist(net.get_lattice(), data, features))
            print(utility.evaluate_som2(net.get_lattice(), data, features))
            plt.pause(0.05)
            print(net.t)
        i += 1

    input()


data, features = mnist_loader('data/mnist.txt')
data = data[:2000]
features = features[:2000]


img_man(data, [15, 15], features, 100, 0.5, False)

