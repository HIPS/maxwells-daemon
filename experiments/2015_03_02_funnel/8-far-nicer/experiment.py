"""A figure to show the nonparametric-ness of the implicit distributions."""

import os
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

import pickle
from collections import defaultdict

from maxwell_d.optimizers import sgd

favblue = (30.0/255.0, 100.0/255.0, 255.0/255.0)

# ------ Problem parameters -------
def nllfun(x):
    """Unnormalized likelihood of a Gaussian evaluated at zero
       having mean and variance given by x[0] and x[1]."""
    mu = x[0]
    logsigma = x[1]
    return logsigma + 0.5*np.sum((mu/(np.exp(logsigma) + 0.01))**2) + 0.1/np.exp(logsigma)
nllfunt = lambda x, t : nllfun(x)
gradfun = grad(nllfunt)

# ------ Variational parameters -------
D = 2
seed = 0
init_scale = 1.4
init_mu = np.array([-2.0, 1.0])
N_iter = 1500
alpha = 0.01

# ------ Plot parameters -------
N_samples = 15
#sample_trails_ix = [int(i) for i in np.floor(np.linspace(0,N_samples, N_samples_trails))][:-1]
N_snapshots = 75
spacing = N_iter / N_snapshots
snapshot_times = range(0,N_iter+1,spacing)
trail_lengths = N_iter

def run():
    all_xs = npr.RandomState(0).randn(N_samples, D) * init_scale + init_mu

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        def callback(x, t, g, v):
            #if i in sample_trails_ix:
            #    results[("trail_x", i)].append(x.copy())
            #results[("likelihood", i)].append(-nllfun(x))
            #if t in snapshot_times:
            results[("all_x", t)].append(x.copy())

        x = sgd(gradfun, x=all_xs[i], iters=N_iter, learn_rate=alpha, decay=0.0, callback=callback)
        callback(x, N_iter, 0.0, 0.0)
    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

xlimits = [-4, 4]
ylimits = [-4.1, 4]


def plot_true_posterior():
    true_posterior_contour_levels = [0.01, 0.2, 1.0, 10.0]

    x = np.linspace(*xlimits, num=200)
    y = np.linspace(*ylimits, num=200)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(0); fig.clf()
    fig.set_size_inches((5,4))
    ax = fig.add_subplot(111)
    zs = np.array([nllfun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, np.exp(-Z), true_posterior_contour_levels, colors='k')
    ax.set_yticks([])
    ax.set_xticks([])
    return ax

def plot():
    print "Plotting results..."
    if not os.path.exists('figures'): os.makedirs('figures')

    with open('results.pkl') as f:
          results = pickle.load(f)

    for ix, t in enumerate(snapshot_times):
        ax = plot_true_posterior()
        #for i in sample_trails_ix:
        #    zipped = zip(*results[("trail_x", i)])
        #    x = zipped[0][t]
        #    y = zipped[1][t]
        coords = np.array(results[("all_x", t)])
        for coord in coords:
            plt.plot(coord[0], coord[1], '*')
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        cur_dir = 'figures/all_trails'
        if not os.path.exists(cur_dir): os.makedirs(cur_dir)
        plt.savefig(cur_dir + "/iter_{}.png".format(ix))
        plt.savefig(cur_dir + "/iter_{}.pdf".format(ix), pad_inches=0.05, bbox_inches='tight')


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
