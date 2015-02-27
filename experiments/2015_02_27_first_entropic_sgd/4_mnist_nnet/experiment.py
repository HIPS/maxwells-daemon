"""First experiment with Hessian-based entropy estimate."""

import numpy as np
from numpy.linalg import norm
import pickle
from collections import defaultdict
from copy import copy
from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import sgd_entropic
from maxwell_d.nn_utils import make_nn_funs
from maxwell_d.data import load_data_subset

# ------ Problem parameters -------
layer_sizes = [784, 500, 10]
batch_size = 200
N_train = 10**3
N_tests = 10**3
# ------ Variational parameters -------
seed = 0
init_scale = 0.05
N_iter = 500
alpha = 0.1 / N_train
# ------ Plot parameters -------
N_samples = 5
N_checkpoints = 50
thin = np.ceil(N_iter/N_checkpoints)

def neg_log_prior(w):
    # return 0.5 * np.dot(w, w) / init_scale**2
    return 0.0

def run():
    (train_images, train_labels),\
    (tests_images, tests_labels) = load_data_subset(N_train, N_tests)
    parser, pred_fun, nllfun, frac_err = make_nn_funs(layer_sizes)
    N_param = len(parser.vect)

    def indexed_loss_fun(w, i_iter):
        rs = RandomState((seed, i, i_iter))
        idxs = rs.randint(N_train, size=batch_size)
        nll = nllfun(w, train_images[idxs], train_labels[idxs]) * N_train
        nlp = neg_log_prior(w)
        return nll + nlp
    gradfun = grad(indexed_loss_fun)

    def callback(x, t, entropy):
        results["entropy_per_dpt"     ].append(entropy / N_train)
        results["minibatch_likelihood"].append(-indexed_loss_fun(x, t))
        results["log_prior_per_dpt"   ].append(-neg_log_prior(x) / N_train)
        if t % thin != 0 and t != N_iter and t != 0: return
        results["iterations"      ].append(t)
        results["train_likelihood"].append(-nllfun(x, train_images, train_labels))
        results["tests_likelihood"].append(-nllfun(x, tests_images, tests_labels))
        results["tests_error"     ].append(frac_err(x, tests_images, tests_labels))
        results["marg_likelihood" ].append(estimate_marginal_likelihood(
            results["train_likelihood"][-1], results["entropy_per_dpt"][-1]))
                                           
        print "Iteration {0:5} Train lik {1:2.4f}  Test lik {2:2.4f}" \
              "  Marg lik {3:2.4f}  Test err {4:2.4f}".format(
                  t, results["train_likelihood"][-1],
                  results["tests_likelihood"][-1],
                  results["marg_likelihood" ][-1],
                  results["tests_error"     ][-1])

    all_results = []
    for i in xrange(N_samples):
        results = defaultdict(list)
        rs = RandomState((seed, i))
        sgd_entropic(gradfun, np.full(N_param, init_scale), N_iter, alpha, rs, callback)
        all_results.append(results)

    return all_results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    with open('results.pkl') as f:
          all_results = pickle.load(f)

    for key in all_results[0]:
        plot_traces_and_mean(all_results, key)

def plot_traces_and_mean(all_results, trace_type, X=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    if X is None:
        X = np.arange(len(results[0][trace_type]))
    for i in xrange(N_samples):
        plt.plot(X, all_results[i][trace_type])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(trace_type)
    ax = fig.add_subplot(212)
    all_Y = [np.array(results[i][trace_type]) for i in range(N_samples)]
    plt.plot(X, sum(all_Y) / float(len(all_Y)))
    plt.savefig(trace_type + '.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
