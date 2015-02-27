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
nllfun = lambda x: 0.5*np.log(2.0*np.pi) + 0.5*np.sum((x-mu)**2)  # Marg lik should be 1
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)
mu = 0.0 # Objective function optimium
D = 1
# ------ Variational parameters -------
seed = 0
init_scale = 3.0
alpha = 0.1
N_iter = 50
# ------ Plot parameters -------
N_samples = 20

def run():
    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        print i,
        def callback(**kwargs):
            for k, v in kwargs.iteritems():
                results[(k, i)].append(copy(v))
            results[("likelihood", i)].append(-nllfun(kwargs['x']))
            results[("x_minus_mu_sq", i)].append((kwargs['x'] - mu)**2)

        rs = RandomState((seed, i))
        sgd_entropic(gradfun, np.full(D, init_scale), N_iter, alpha, rs, callback)

    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    with open('results.pkl') as f:
          results = pickle.load(f)

    iters = results[('t', 0)]
    for i in xrange(N_samples):
        results[('marginal_likelihood', i)] = estimate_marginal_likelihood(
            results[("likelihood", i)], np.array(results[("entropy", i)])[iters])

    plot_traces_and_mean(results, 'entropy')
    plot_traces_and_mean(results, 'x_minus_mu_sq')
    plot_traces_and_mean(results, 'likelihood')
    plot_traces_and_mean(results, 'x')
    plot_traces_and_mean(results, 'marginal_likelihood', X=iters)

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

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
