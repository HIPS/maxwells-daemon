"""Some simple diagnostic tests to check whether entropic descent is working."""

import numpy as np
import pickle
from collections import defaultdict
from numpy.linalg import norm

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import entropic_descent2

# ------ Problem parameters -------
nllfun = lambda x: 0.5*D*np.log(2.0*np.pi) + 0.5*np.sum((x-mu)**2)  # Marg lik should be 1
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)
# ------ Variational parameters -------

D = 100
seed = 2
init_scale = 3.0
mu = 2.0
epsilon = 1.0
gamma = 0.3
N_iter = 1800
alpha = 0.1
# ------ Plot parameters -------
N_samples = 100

def run():
    x_init_scale = np.full(D, init_scale)
    # annealing_schedule = np.linspace(0,1,N_iter)
    annealing_schedule = np.concatenate((np.zeros(N_iter/3),
                                         np.linspace(0, 1, N_iter/3),
                                         np.ones(N_iter/3)))
    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        def callback(x, t, v, entropy):
            #results[("x", i)].append(x.copy())    # Replace this with a loop over kwargs?
            results[("entropy", i)].append(entropy)
            #results[("velocity", i)].append(v)
            results[("likelihood", i)].append(-nllfun(x))

        rs = RandomState((seed, i))
        x, entropy = entropic_descent2(gradfun, callback=callback, x_scale=x_init_scale,
                                       epsilon=epsilon, gamma=gamma, alpha=alpha,
                                       annealing_schedule=annealing_schedule, rs=rs)
    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
          results = pickle.load(f)

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[("entropy", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("entropy", i)][t] for i in xrange(N_samples)])
              for t in xrange(N_iter)])
    plt.savefig("entropy.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[("likelihood", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("likelihood", i)][t] for i in xrange(N_samples)]) for t in xrange(N_iter)])
    plt.savefig("likelihoods.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot([estimate_marginal_likelihood(results[("likelihood", i)][t],
                                               results[("entropy", i)][t])
                  for t in xrange(N_iter)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([estimate_marginal_likelihood(results[("likelihood", i)][t],
                                                    results[("entropy", i)][t])
                       for i in xrange(N_samples)])
              for t in xrange(N_iter)])
    plt.savefig("marginal_likelihoods.png")

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
