"""A simple phase diagram figure."""

import numpy as np
import pickle
from collections import defaultdict
from numpy.linalg import norm

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import entropic_descent2

# ------ Problem parameters -------
mu = 2.0
nllfun = lambda x: 0.5*np.log(2.0*np.pi) + 0.5*np.sum((x-mu)**2)  # Marg lik should be 1
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)

# ------ Variational parameters -------
D = 1
seed = 0
init_scale = 3.0
epsilon = 1.0
gamma = 0.05
N_iter = 1000
alpha = 0.05

# ------ Plot parameters -------
N_samples_trails = 10
N_samples = 1000
N_snapshots = 4
spacing = N_iter / N_snapshots
snapshot_times = range(0,N_iter+1,spacing)
trail_lengths = 15
kernel_width = 0.5
num_contours = 4

def run():
    x_init_scale = np.full(D, init_scale)
    annealing_schedule = np.linspace(0,1,N_iter)

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        def callback(x, t, v, entropy):
            if i < N_samples_trails:
                results[("trail_x", i)].append(x.copy())
                results[("trail_v", i)].append(v.copy())
            results[("entropy", i)].append(entropy)
            results[("likelihood", i)].append(-nllfun(x))
            if t in snapshot_times:
                results[("all_x", t)].append(x.copy())
                results[("all_v", t)].append(v.copy())

        rs = RandomState((seed, i))
        entropic_descent2(gradfun, callback=callback, x_scale=x_init_scale,
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

    # ------ contour plots --------
    xlimits = [-init_scale * 2, init_scale * 2]
    ylimits = [-2, 2]

    def kernel(mux, muy, x,y):
        return np.exp(-((mux - x)/kernel_width)**2 - ((muy - y)/kernel_width)**2)

    def smoothed_density(xp, yp, xq, yq):
        densities = np.zeros(xq.shape[0])
        for i in xrange(len(xp)):
            densities += kernel(xp[i], yp[i], xq, yq)
        return densities

    for t in snapshot_times:
        print "Plotting contour at time {0}".format(t)
        fig = plt.figure(0); fig.clf()
        ax = fig.add_subplot(111)
        x = np.linspace(*xlimits, num=100)
        y = np.linspace(*ylimits, num=100)
        X, Y = np.meshgrid(x, y)
        zs = smoothed_density(results[("all_x", t)], results[("all_v", t)], np.ravel(X),np.ravel(Y))
        Z = zs.reshape(X.shape)
        CS = plt.contour(X, Y, Z, num_contours, colors='k')

        # Plot trails on top.
        start_t = max(0, t-trail_lengths)
        for i in xrange(N_samples_trails):
            plt.plot(results[("trail_x", i)][start_t:t+1], results[("trail_v", i)][start_t:t+1])
        for i in xrange(N_samples_trails):
            plt.plot(results[("trail_x", i)][t], results[("trail_v", i)][t], '*k')
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.set_yticks([0.0])
        ax.set_xticks([0.0, mu])
        fig.set_size_inches((2,2))
        plt.savefig("seqs/densities_{0}.png".format(t))
        plt.savefig("seqs/densities_{0}.pdf".format(t), pad_inches=0.0, bbox_inches='tight')

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples_trails):
        plt.plot(results[("trail_v", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("trail_v", i)][t]**2 for i in xrange(N_samples)]) for t in xrange(N_iter)])
    ax.set_title("Mean squared velocity")
    plt.savefig("v_paths.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    for i in xrange(N_samples_trails):
        plt.plot(results[("trail_x", i)], results[("trail_v", i)])
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
    #results = run()
    #with open('results.pkl', 'w') as f:
    #    pickle.dump(results, f, 1)
    plot()
