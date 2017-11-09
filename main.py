import numpy as np
from som import *
from data_reader import *
from matplotlib import pyplot as plt
import time
from multiprocessing import Process, Value, Array
import sys



def worker(tId, best, case, target, run_counter, proc_activity):
    np.random.seed(tId)
    while True:
        proc_activity[tId] = 1
        run_counter.value += 1
        if best[0] == tId and best[3] < 20:
            start = time.time()
            score = tsp_man(case, best[5], best[6], best[7], best[8], best[9], False)
            if score < best[1]: best[1] = score
            if score > best[2]: best[2] = score
            best[4] = (best[4]*best[3]+time.time()-start)/(best[3]+1)
            best[3] += 1
        else:
            start = time.time()
            r = np.random.rand(5) + .5
            nf = best[5]*r[0]
            lr = best[6]*r[1]
            lam_n = best[7]*r[2]
            lam_lr = best[8]*r[3]
            sig = best[9]*r[4]
            score = tsp_man(case, nf, lr, lam_n, lam_lr, sig, False)
            if score < best[1] or (round(score) == target and time.time()-start < best(4)):
                best[0] = tId
                best[1] = score
                best[2] = score
                best[3] = 1
                best[4] = time.time()-start
                best[5] = nf
                best[6] = lr
                best[7] = lam_n
                best[8] = lam_lr
                best[9] = sig



def genetic_brute(nProcs=4):

    proc_times = [0] * nProcs
    procs = [None] * nProcs
    case = 1
    optimum = 7542
    run_counter = Value('i', 0)
    proc_activity = Array('i', nProcs)

    """
    0: ownerId, 1: best_score, 2: worst_score, 3:times_tested, 4: execution_time
    5: node_factor, 6: lr0, 7: lambda_n, 8:lambda_lr, 9: sigma
    """
    current_best = Array('d', [-1.0, 999999999.9, 999999999.9, 0.0, 99999.9,
                                    2.5, 0.1, 10000.0, 100000.0, 30.0])


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
            print('Process {0} has the best run so far'.format(int(current_best[0])))
            print('Best score: {0}\nWorst score: {1}'.format(round(current_best[1]*100)/100, round(current_best[2]*100)/100))
            print('Average time: {0} over {1} runs'. format(round(current_best[4]*100)/100, int(current_best[3])))
            print('Variables:\n    Node Factor: {0}\n    Learning Rate: {1}'.format(current_best[5], current_best[6]))
            print('    Lambda_n: {0}\n    Lambda_lr: {1}'.format(current_best[7], current_best[8]))
            print('    Sigma: {}'.format(current_best[9]))
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

if len(sys.argv) == 1:
    genetic_brute(4)
else:
    try:
        genetic_brute(int(sys.argv[1]))
    except:
        print("Holy crab sticks! We be crashing! \\0/")
        sys.exit()

