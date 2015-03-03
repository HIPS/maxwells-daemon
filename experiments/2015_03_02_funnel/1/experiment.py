"""A figure to show the nonparametric-ness of the implicit distributions."""

import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import sgd_entropic

# ------ Problem parameters -------
def nllfun(x):
    """Unnormalized likelihood of a Gaussian evaluated at zero
       having mean and variance given by x[0] and x[1]."""
    mu = x[0]
    logsigma = x[1]
    return logsigma + 0.5*np.sum((mu/(np.exp(logsigma) + 0.01))**2)
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)

# ------ Variational parameters -------
D = 2
seed = 0
init_scale = 1.25
init_mu = np.array([-0.8, 0.5])
N_iter = 20
alpha = 0.05

# ------ Plot parameters -------
N_samples_trails = 50
N_samples = 500
N_snapshots = 2
spacing = N_iter / N_snapshots
snapshot_times = range(0,N_iter+1,spacing)
trail_lengths = N_iter
kernel_width = 0.1

def run():
    x_init_scale = np.full(D, init_scale)

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        def callback(x, t, entropy):
            if i < N_samples_trails:
                results[("trail_x", i)].append(x.copy())
            results[("entropy", i)].append(entropy)
            results[("likelihood", i)].append(-nllfun(x))
            if t in snapshot_times:
                results[("all_x", t)].append(x.copy())

        rs = RandomState((seed, i))
        x, entropy = sgd_entropic(gradfun, x_scale=x_init_scale, N_iter=N_iter, learn_rate=alpha,
                                  rs=rs, callback=callback, approx=False, mu=init_mu)
        callback(x, N_iter, entropy)
    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
          results = pickle.load(f)

    # ------ contour plots --------
    xlimits = [-4, 4]
    ylimits = [-4, 4]

    tp_levels = [0.2, 1.0, 10.0]   # Contour levels of true posterior
    ap_levels = [0.2]   # Contour levels of true posterior

    def kernel(mux, muy, x,y):
        kernel_width_x = np.std(x) * kernel_width
        kernel_width_y = np.std(y) * kernel_width
        return np.exp(-((mux - x)/kernel_width_x)**2 - ((muy - y)/kernel_width_y)**2)/len(x)

    def smoothed_density(xp, yp, xq, yq):
        densities = np.zeros(len(xq))
        for i in xrange(len(xp)):
            densities += kernel(xp[i], yp[i], xq, yq)
        return densities

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)

    x = np.linspace(*xlimits, num=100)
    y = np.linspace(*ylimits, num=100)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    zs = np.array([nllfun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    CS = plt.contour(X, Y, np.exp(-Z), tp_levels, colors='k', ls='--')
    #plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig("true_density.png")

    for t in snapshot_times:
        print "Plotting contour at time {0}".format(t)
        zipped = zip(*results[("all_x", t)])
        xs = list(zipped[0])
        ys = list(zipped[1])
        zs = smoothed_density(xs, ys, np.ravel(X), np.ravel(Y))
        Z = zs.reshape(X.shape)
        print "number of nans: ", np.sum(np.isnan(Z))
        CS = plt.contour(X, Y, Z, 1, colors='r')

    # Plot trails on top.
    for i in xrange(N_samples_trails):
        zipped = zip(*results[("trail_x", i)])
        xs = list(zipped[0])
        ys = list(zipped[1])
        plt.plot(xs, ys, 'b', alpha=0.5)

    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_yticks([0.0])
    ax.set_xticks([0.0])
    fig.set_size_inches((5,4))
    plt.savefig("dists.png")
    plt.savefig("dists.pdf".format(t), pad_inches=0.05, bbox_inches='tight')

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
    plt.plot([np.mean([results[("likelihood", i)][t] for i in xrange(N_samples)])
              for t in xrange(N_iter)])
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
