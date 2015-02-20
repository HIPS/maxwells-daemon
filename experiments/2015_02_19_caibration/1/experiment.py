"""Some simple diagnostic tests to check whether entropic descent is working."""

import numpy as np
import pickle
from collections import defaultdict
from numpy.linalg import norm

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import entropic_descent2

# ------ Problem parameters -------
nllfun = lambda x: -0.5 * np.log(2.0*np.pi) + 0.5*np.sum(x**2)  # Marg lik should be 1
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)

# ------ Variational parameters -------
D = 1
seed = 0
init_variance = 10.0
x_init_scale = np.full(D, init_variance)
epsilon = 0.11
gamma = 0.01
N_iter = 200

# ------ Plot parameters -------
N_samples = 5

def run():
    print "Running experiment..."

    results = defaultdict(list)
    for i in xrange(N_samples):
        def callback(x, t, g, v, entropy):
            results[("x", i, t)] = x.copy()    # Replace this with a loop over kwargs?
            results[("entropy", i, t)] = entropy
            results[("velocity", i, t)] = v
            results[("likelihood", i, t)] = -nllfun(x)

        rs = RandomState((seed, i))
        x, entropy = entropic_descent2(gradfun, callback=callback, x_scale=x_init_scale,
                                       epsilon=epsilon, gamma=gamma, iters=N_iter, rs=rs)
        results[("x", i, N_iter)] = x
        results[("entropy", i, N_iter)] = x

    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
          results = pickle.load(f)

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    for i in xrange(N_samples):
        plt.plot([results[("x", i, t)] for t in xrange(N_iter)])
    plt.savefig("paths.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    for i in xrange(N_samples):
        plt.plot([results[("entropy", i, t)] for t in xrange(N_iter)])
    plt.savefig("entropy.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    for i in xrange(N_samples):
        plt.plot([results[("likelihood", i, t)] for t in xrange(N_iter)])
    plt.savefig("likelihoods.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    for i in xrange(N_samples):
        plt.plot([estimate_marginal_likelihood(results[("likelihood", i, t)],
                                               results[("entropy", i, t)]) for t in xrange(N_iter)])
    plt.savefig("marginal_likelihoods.png")


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
