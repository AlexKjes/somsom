import numpy as np
from matplotlib.colors import to_rgb



colors = ['red', 'orange', 'lime', 'blue', 'gray', 'yellow', 'cyan', 'green', 'pink', 'magenta']

def chessboard_distance(x):
    return np.max(x)


def manhattan_distance(x):
    return np.sum(x)


def square_euclidean_distance(x):
    return np.sum(np.square(x))


def euclidean_distance(x):
    return np.sqrt(square_euclidean_distance(x))


def generate_distance_matrix(shape, wrap=True, distance_fn=manhattan_distance):
    n_nodes = np.prod(shape)
    ret = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        md_i = np.unravel_index(i, shape)
        for j in range(n_nodes):
            distances = np.abs(np.subtract(md_i, np.unravel_index(j, shape)))
            # neighbourhood matrix wraparound
            if wrap:
                for k, d in enumerate(shape):
                    if distances[k] > shape[k ] /2:
                        distances[k] = shape[k] - distances[k]
            distance = distance_fn(distances)
            ret[i][j] = distance
    return ret


def total_distance(weights):

    td = 0
    for i in range(weights.shape[0]-1):
        td += np.sqrt(np.sum(np.square(weights[i] - weights[i+1])))
    td += np.sqrt(np.sum(np.square(weights[-1] - weights[0])))
    return td


def weight_circle(data, n_nodes, radius_factor=1, translate_x=1, translate_y=1):
    dim_max = np.max(data, 0)
    i_max = np.argmax(dim_max)
    dim_min = np.min(data, 0)
    dim_center = np.mean(data, 0)
    dim_diam = (dim_max - dim_min)
    dim_radius = dim_diam / 2
    scale = (dim_max-dim_min)/(n_nodes/2)

    ret = np.zeros([n_nodes, data.shape[1]])

    for i, x in enumerate(np.concatenate((np.arange(-dim_radius[i_max], dim_radius[i_max], scale[i_max]),
                                         np.arange(dim_radius[i_max], -dim_radius[i_max], -scale[i_max])))[:n_nodes]):

        x *= radius_factor
        y = np.sqrt((dim_radius[i_max]*radius_factor)**2-x**2)
        x += dim_center[0] * translate_x

        if i <= n_nodes/2:
            y += dim_center[1] * translate_y
            ret[i] = [x, y]
        else:
            y *= -1
            y += dim_center[1] * translate_y
            ret[i] = [x, y]

    return ret


def generate_contour_map(lattice):

    ret = np.zeros(lattice.shape[:2])

    for y, r in enumerate(lattice):
        for x, _ in enumerate(r):

            n = []
            if x > 0 < y: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y - 1, x - 1]))))
            if 0 < y: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y - 1, x]))))
            if 0 < y and x < lattice.shape[1] - 1: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y - 1, x + 1]))))
            if x > 0: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y, x - 1]))))
            if x < lattice.shape[1] - 1: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y, x + 1]))))
            if x > 0 and y < lattice.shape[0] - 1: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y + 1, x - 1]))))
            if y < lattice.shape[0] - 1: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y + 1, x]))))
            if y < lattice.shape[0] - 1 and x < lattice.shape[1] - 1: n.append(np.sqrt(np.sum(np.square(lattice[y, x] - lattice[y + 1, x + 1]))))
            ret[y, x] = np.mean(n)

    mi = np.min(ret)
    ret = -((ret-mi)/(np.max(ret)-mi))+1

    return ret


def map_mnist(lattice, data, features):
    cm, ci = class_map(lattice, data, features)

    return np.array([[np.concatenate((to_rgb(colors[v]), [i])) for v, i in zip(mr, ir)] for mr, ir in zip(cm, ci)])


def class_map(lattice, data, features):

    lh = lattice_histogram(lattice, data, features)

    ret_c = np.zeros((lattice.shape[0], lattice.shape[0]), np.int64)
    ret_i = np.zeros((lattice.shape[0], lattice.shape[0]), np.float64)

    for y, r in enumerate(lh):
        for x, v in enumerate(r):
            ret_c[y, x] = np.argmax(lh[y, x])
            ret_i[y, x] = np.max(lh[y, x])

    ret_i = np.log(ret_i, where=ret_i>0)
    return ret_c, ret_i/np.max(ret_i)


def lattice_histogram(lattice, data, features):
    n_unique_features = np.unique(features).shape[0]
    lh = np.zeros((lattice.shape[0], lattice.shape[1], n_unique_features))
    for d, f in zip(data, features):
        bmu_index = np.unravel_index(np.argmin(np.sum(np.abs(d - lattice), 2)), lattice.shape[:2])
        lh[bmu_index][f] += 1

    return lh

def evaluate_som(lattice, data, features, test_data, test_features):

    cm = class_map(lattice, data, features)
    correct = 0
    for td, tf in zip(test_data, test_features):
        bmu_index = np.unravel_index(np.argmin(np.sum(np.abs(td-lattice), 2)), lattice.shape[:2])
        if cm[bmu_index] == tf:
            correct += 1

    return correct/test_data.shape[0]


def evaluate_som2(lattice, test_data, features):

    n_unique_features = np.unique(features).shape[0]
    blabla = np.zeros((lattice.shape[0], lattice.shape[1], n_unique_features))
    for d, f in zip(test_data, features):
        bmu_index = np.unravel_index(np.argmin(np.sum(np.abs(d-lattice), 2)), lattice.shape[:2])
        blabla[bmu_index][f] += 1

    return np.sum(np.max(blabla, 2)) / np.sum(blabla)
