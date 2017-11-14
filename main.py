import numpy as np
from som import *
from data_reader import *
from matplotlib import pyplot as plt
import time
from multiprocessing import Process, Value, Array
import sys
import test


OWNER = 0
BEST_SCORE = 1
WORST_SCORE = 2
AVERAGE_SCORE = 3
TIMES_TESTED = 4
AVERAGE_TIME = 5

NODE_FACTOR = 6
LEARNING_RATE = 7
LAMBDA_N = 8
LAMBDA_LR = 9
SIGMA = 10

RADIUS_FACTOR = 11
TRANSLATE_X = 12
TRANSLATE_Y = 13



def worker(tId, best, case, target, run_counter, proc_activity):
    local = [v for v in best]
    np.random.seed(tId)
    while True:
        proc_activity[tId] = 1
        run_counter.value += 1
        if best[OWNER] == tId and best[TIMES_TESTED] < 40:
            start = time.time()
            score = tsp_man(case, best[NODE_FACTOR], best[LEARNING_RATE], best[LAMBDA_N], best[LAMBDA_LR], best[SIGMA],
                            best[RADIUS_FACTOR], best[TRANSLATE_X], best[TRANSLATE_Y], visualize=False)
            if score < best[BEST_SCORE]: best[BEST_SCORE] = score
            if score > best[WORST_SCORE]: best[WORST_SCORE] = score
            best[AVERAGE_TIME] = (best[AVERAGE_TIME]*best[TIMES_TESTED]+time.time()-start)/(best[TIMES_TESTED]+1)
            best[AVERAGE_SCORE] = (best[AVERAGE_SCORE]*best[TIMES_TESTED]+score)/(best[TIMES_TESTED]+1)
            best[TIMES_TESTED] += 1
        elif local[BEST_SCORE] < best[WORST_SCORE] and np.abs(local[AVERAGE_SCORE]-target) < np.abs(best[AVERAGE_SCORE]-target):
            start = time.time()
            score = tsp_man(case, local[NODE_FACTOR], local[LEARNING_RATE], local[LAMBDA_N], local[LAMBDA_LR], 
                            local[SIGMA], visualize=False)
            local[AVERAGE_SCORE] = (local[AVERAGE_SCORE]*local[TIMES_TESTED]+score)/(local[TIMES_TESTED]+1)
            if score < local[BEST_SCORE]: local[BEST_SCORE] = score
            if score > local[WORST_SCORE]: local[WORST_SCORE] = score
            local[AVERAGE_TIME] = (local[AVERAGE_TIME]*local[TIMES_TESTED]+time.time()-start)/(local[TIMES_TESTED]+1)
            local[TIMES_TESTED] += 1
            if local[TIMES_TESTED] > 6 and [local[WORST_SCORE]<best[WORST_SCORE]]:# or (np.abs(local[AVERAGE_SCORE]-target) < np.abs(best[AVERAGE_SCORE]-target)):
                for i, v in enumerate(local):
                    best[i] = v
                local[AVERAGE_SCORE] = 9999999
                local[BEST_SCORE] = 9999999
                local[WORST_SCORE] = 9999999
        else:
            start = time.time()
            r = np.random.rand(8) + .5
            nf = best[NODE_FACTOR]*r[0]
            nf = nf if nf > 2 else 2.0
            lr = best[LEARNING_RATE]*r[1]
            lam_n = best[LAMBDA_N]*r[2]
            lam_lr = best[LAMBDA_LR]*r[3]
            sig = best[SIGMA]*r[4]
            rf = best[RADIUS_FACTOR]*r[5]
            tx = best[TRANSLATE_X]*r[6]
            ty = best[TRANSLATE_Y]*r[7]
            score = tsp_man(case, nf, lr, lam_n, lam_lr, sig, rf, tx, ty, visualize=False)
            if np.abs(score-target) < np.abs(best[BEST_SCORE] - target) or (round(score) == target and time.time()-start < best(4)):
                local[OWNER] = tId
                local[BEST_SCORE] = score
                local[WORST_SCORE] = score
                local[AVERAGE_SCORE] = score
                local[AVERAGE_TIME] = 1
                local[AVERAGE_TIME] = time.time()-start
                local[NODE_FACTOR] = nf
                local[LEARNING_RATE] = lr
                local[LAMBDA_N] = lam_n
                local[LAMBDA_LR] = lam_lr
                local[SIGMA] = sig
                local[RADIUS_FACTOR] = rf
                local[TRANSLATE_X] = tx
                local[TRANSLATE_Y] = ty


def genetic_brute(nProcs=4):

    proc_times = [0] * nProcs
    procs = [None] * nProcs
    case = 1
    optimum = 7542
    run_counter = Value('i', 0)
    proc_activity = Array('i', nProcs)

    """
    0: ownerId, 1: best_score, 2: worst_score, 3: average_ score, 4:times_tested, 5: execution_time
    6: node_factor, 7: lr0, 8: lambda_n, 9:lambda_lr, 10: sigma, 11: radius_factor, 12: translate_x, 13: translate_y
    """
    current_best = Array('d', [-1.0, 999999999.9, 999999999.9, 99999999.9, 0.0, 99999.9,
                                    2.5, 0.1, 10000.0, 100000.0, 30.0, 1.0, 1.0, 1.0])


    for i in range(nProcs):
        procs[i] = (Process(target=worker, args=(i, current_best, case, optimum, run_counter, proc_activity)))
        proc_times[i] = (time.time())
        procs[i].start()

    log_timer = time.time()

    while True:
        iter_time = time.time()
        if time.time()-log_timer > 30:
            log_timer = iter_time
            print('---------------------------------------------')
            print('{0} Self organizing maps trained'.format(run_counter.value))
            print('Process {0} has the best run so far'.format(int(current_best[OWNER])))
            print('Best score: {0}\nWorst score: {1}'.format(round(current_best[BEST_SCORE]*100)/100, round(current_best[WORST_SCORE]*100)/100))
            print('Average score: {}'.format((round(current_best[AVERAGE_SCORE]*100)/100)))
            print('Average time: {0} over {1} runs'. format(round(current_best[AVERAGE_TIME]*100)/100, int(current_best[TIMES_TESTED])))
            print('Variables:\n    Node Factor: {0}\n    Learning Rate: {1}'.format(current_best[NODE_FACTOR], current_best[LEARNING_RATE]))
            print('    Lambda_n: {0}\n    Lambda_lr: {1}'.format(current_best[LAMBDA_N], current_best[LAMBDA_LR]))
            print('    Sigma: {0}\n    Radius factor: {1}'.format(current_best[SIGMA], current_best[RADIUS_FACTOR]))
            print('    Translate x: {0}\n    Translate y: {1}'.format(current_best[TRANSLATE_X], current_best[TRANSLATE_Y]))
            print('cptString: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'.format(current_best[NODE_FACTOR], current_best[LEARNING_RATE],
                                                                             current_best[LAMBDA_N], current_best[LAMBDA_LR],
                                                                             current_best[SIGMA], current_best[RADIUS_FACTOR],
                                                                             current_best[TRANSLATE_X], current_best[TRANSLATE_Y]))
            print('---------------------------------------------')

        for i, (t, p) in enumerate(zip(proc_times, procs)):
            if iter_time-proc_times[i] > 120 and proc_activity[i]:
                print("Process {} timed out and is killed".format(i))
                p.terminate()
                procs[i] = Process(target=worker, args=(i, current_best, case, optimum, run_counter, proc_activity))
                proc_times[i] = iter_time
            else:
                proc_activity = [0]
                proc_times[i] = iter_time

        time.sleep(2)


"""
if len(sys.argv) == 1:
    genetic_brute(4)
else:
    try:
        genetic_brute(int(sys.argv[1]))
    except:
        print("Holy crab sticks! We be crashing! \\0/")
        sys.exit()

"""


#tsp_man(1, 3, 0.4, 100, 200, 100, 0.1, 0.7, 2.1)


#img_man(data, [1, 100], features=features, k=5, lr0=0.05, lambda_n=1000, lambda_lr=1000, sigma=10)
"""
colors = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])
"""

data, features = mnist_loader('data/mnist.txt')

test_data = data[6000:]
test_features = features[6000:]

data = data[:1000]
features = features[:1000]

img_man(data, [20, 20], k=5, lr0=0.2, lambda_n=5000, lambda_lr=10000, sigma=15, features=features,
        test_data=test_data, test_features=test_features)
input()



"""
data, features = mnist_loader('data/mnist.txt')
data = data[:100]
features = features[:100]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Training inputs for RGBcolors
colors = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

print(features)
print(data.shape)

# Train a 20x30 SOM with 400 iterations
som = test.SOM(20, 20, 784, 400)
som.train(data)

# Get output grid
image_grid = som.get_centroids()
print(len(image_grid[0]))
# Map colours to their closest neurons
mapped = som.map_vects(features)
print()
# Plot
plt.imshow(image_grid)
plt.title('Color SOM')
for i, (m, f) in enumerate(zip(mapped, features)):
    plt.text(m[1], m[0], labels[f], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
"""