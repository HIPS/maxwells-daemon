"""Some simple diagnostic tests to check whether entropic descent is working."""

import numpy as np
import pickle
from collections import defaultdict
from numpy.linalg import norm

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import aed3_anneal

# ------ Problem parameters -------
nllfun = lambda x: 0.5*np.log(2.0*np.pi) + 0.5*np.sum((x-mu)**2)  # Marg lik should be 1
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)
# ------ Variational parameters -------
D = 1
seed = 2
init_scale = 3.0
mu = 10.0
epsilon = 0.1
gamma = 0.3
N_iter = 1800
alpha = 0.1
init_log_decay=np.log(1.0)
decay_learn_rate=0.0

x_init_scale = np.full(D, init_scale)
# annealing_schedule = np.linspace(0,1,N_iter)
annealing_schedule = np.concatenate((np.zeros(N_iter/3),
                                     np.linspace(0, 1, N_iter/3),
                                     np.ones(N_iter/3)))

# ------ Plot parameters -------
N_samples = 10

def run():
    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        print i,
        def callback(x, t, v, entropy, log_decay):
            results[("x", i)].append(x.copy())    # Replace this with a loop over kwargs?
            results[("entropy", i)].append(entropy)
            results[("velocity", i)].append(v)
            results[("log decay", i)].append(log_decay[0])
            results[("likelihood", i)].append(-nllfun(x))

        rs = RandomState((seed, i))
        aed3_anneal(gradfun, x_scale=x_init_scale, callback=callback, learn_rate=epsilon,
                    init_log_decay=init_log_decay, decay_learn_rate=decay_learn_rate, rs=rs,
                    annealing_schedule=annealing_schedule)
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
        plt.plot(results[("x", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("x", i)][t] for i in xrange(N_samples)]) for t in xrange(N_iter)])
    plt.savefig("x_paths.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[("velocity", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("velocity", i)][t]**2 for i in xrange(N_samples)]) for t in xrange(N_iter)])
    ax.set_title("Mean squared velocity")
    plt.savefig("v_paths.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    for i in xrange(N_samples):
        plt.plot(results[("x", i)], results[("velocity", i)])
    plt.savefig("trajectories.png")

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
        plt.plot(results[("log decay", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("log decay", i)][t] for i in xrange(N_samples)])
              for t in xrange(N_iter)])
    plt.savefig("log_decay.png")

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
