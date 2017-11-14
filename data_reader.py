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
        print(coords)
        ret[i][0] = float(coords[0])
        ret[i][1] = float(coords[1][:-1])

    return ret


def mnist_loader(fname):
    lines = []
    with open(fname, 'r') as f:
        for l in f:
            lines.append(l)

    for i, l in enumerate(lines):
        ll = []
        for p in l.split(','):
            ll.append(float(p))
        lines[i] = ll
    ret = np.array(lines)
    return ret[:, :-1], ret[:, -1].astype(np.int8)
