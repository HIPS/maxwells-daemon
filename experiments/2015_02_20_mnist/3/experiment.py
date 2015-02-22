"""First real experiment - how well do we do on MNIST?"""

import numpy as np
from numpy.linalg import norm
import pickle
from collections import defaultdict

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import entropic_descent2
from maxwell_d.nn_utils import make_nn_funs
from maxwell_d.data import load_data_subset

# ------ Problem parameters -------
layer_sizes = [784, 10]
batch_size = 200
N_train = 10**3
N_tests = 10**3

# ------ Variational parameters -------
seed = 0
init_scale = 1.0
L2_per_dpt = 1 / init_scale **2 / N_train
epsilon = 0.1 / N_train
gamma = 0.1
N_iter = 2000
alpha = 0.1
annealing_schedule = np.concatenate((np.zeros(N_iter/5),
                                     np.linspace(0, 1, 3 * N_iter/5),
                                     np.ones(N_iter/5)))
# ------ Plot parameters -------
N_samples = 3
N_checkpoints = 30
thin = np.ceil(N_iter/N_checkpoints)

def run():
    (train_images, train_labels),\
    (tests_images, tests_labels) = load_data_subset(N_train, N_tests)
    parser, pred_fun, nllfun, frac_err = make_nn_funs(layer_sizes, L2_per_dpt)
    N_param = len(parser.vect)

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        x_init_scale = np.full(N_param, init_scale)

        def indexed_loss_fun(w, i_iter):
            rs = RandomState((seed, i, i_iter))
            idxs = rs.randint(N_train, size=batch_size)
            return nllfun(w, train_images[idxs], train_labels[idxs]) * N_train
        gradfun = grad(indexed_loss_fun)

        def callback(x, t, v, entropy):
            results[("entropy", i)].append(entropy / N_train)
            results[("v_norm", i)].append(norm(v) / np.sqrt(N_param))
            results[("minibatch_likelihood", i)].append(-indexed_loss_fun(x, t))
            if t % thin != 0 and t != N_iter and t != 0: return
            results[('iterations', i)].append(t)
            results[("train_likelihood", i)].append(-nllfun(x, train_images, train_labels))
            results[("tests_likelihood", i)].append(-nllfun(x, tests_images, tests_labels))
            results[("tests_error", i)].append(frac_err(x, tests_images, tests_labels))
            print "Iteration {0:5} Train likelihood {1:2.4f}  Test likelihood {2:2.4f}" \
                  "  Test Err {3:2.4f}".format(t, results[("train_likelihood", i)][-1],
                                                  results[("tests_likelihood", i)][-1],
                                                  results[("tests_error",      i)][-1])
        rs = RandomState((seed, i))
        entropic_descent2(gradfun, callback=callback, x_scale=x_init_scale,
                          epsilon=epsilon, gamma=gamma, alpha=alpha,
                          annealing_schedule=annealing_schedule, rs=rs)
    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot_traces_and_mean(results, trace_type, X=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    if X is None:
        X = np.arange(len(results[(trace_type, 0)]))
    for i in xrange(N_samples):
        plt.plot(X, results[(trace_type, i)])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(trace_type)
    ax = fig.add_subplot(212)
    all_Y = [np.array(results[(trace_type, i)]) for i in range(N_samples)]
    plt.plot(X, sum(all_Y) / float(len(all_Y)))
    plt.savefig(trace_type + '.png')

def plot():
    print "Plotting results..."
    with open('results.pkl') as f:
          results = pickle.load(f)
    import matplotlib.pyplot as plt

    iters = results[('iterations', 0)]
    for i in xrange(N_samples):
        results[('marginal_likelihood', i)] = estimate_marginal_likelihood(
            results[("train_likelihood", i)],
            np.array(results[("entropy", i)])[iters])

    plot_traces_and_mean(results, 'entropy')
    plot_traces_and_mean(results, 'v_norm')
    plot_traces_and_mean(results, 'minibatch_likelihood')
    plot_traces_and_mean(results, 'tests_likelihood', X=iters)
    plot_traces_and_mean(results, 'train_likelihood', X=iters)
    plot_traces_and_mean(results, 'tests_error',      X=iters)
    plot_traces_and_mean(results, 'marginal_likelihood', X=iters)

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
