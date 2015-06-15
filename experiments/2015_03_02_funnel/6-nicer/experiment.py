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


plot_trails = False

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
init_scale = 2.5 / 3
init_mu = np.array([-1.0, 1.0])
N_iter = 500
alpha = 0.01

# ------ Plot parameters -------
N_samples = 250
N_samples_trails = 5
sample_trails_ix = [int(i) for i in np.floor(np.linspace(0,N_samples, N_samples_trails))][:-1]
N_snapshots = 10
spacing = N_iter / N_snapshots
snapshot_times = range(0,N_iter+1,spacing)
trail_lengths = N_iter
kernel_width = 0.1

num_rings = 3

def make_circle(N=100):
    th = np.linspace(0, 2 * np.pi, N)
    return np.concatenate((np.cos(th)[None, :], np.sin(th)[None, :]), axis=0).T

def run():
    all_xs = np.concatenate([make_circle(N_samples) * init_scale*(i+1) + init_mu for i in range(num_rings)])

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples * num_rings):
        def callback(x, t, g, v):
            if i in sample_trails_ix:
                results[("trail_x", i)].append(x.copy())
            results[("likelihood", i)].append(-nllfun(x))
            if t in snapshot_times:
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

    ax = plot_true_posterior()
    plt.savefig("figures/true_posterior.png")
    plt.savefig("figures/true_posterior.pdf", pad_inches=0.05, bbox_inches='tight')

    if plot_trails:
        for i in sample_trails_ix:
            print "Plotting trail {0}".format(i)
            zipped = zip(*results[("trail_x", i)])
            xs = [zipped[0][t] for t in snapshot_times]
            ys = [zipped[1][t] for t in snapshot_times]
            for ix, (x,y) in enumerate(zip(xs,ys)):
                ax = plot_true_posterior()
                plt.plot(x, y, '*')
                ax.set_xlim(xlimits)
                ax.set_ylim(ylimits)
                cur_dir = 'figures/trails_{}'.format(i)
                if not os.path.exists(cur_dir): os.makedirs(cur_dir)
                plt.savefig(cur_dir + "/iter_{}.png".format(ix))
                plt.savefig(cur_dir + "/iter_{}.pdf".format(ix), pad_inches=0.05, bbox_inches='tight')

    for i, t in enumerate(snapshot_times):
        print "Plotting contour at time {0}".format(t)
        ax = plot_true_posterior()
        zipped = zip(*results[("all_x", t)])
        xs = list(zipped[0])
        ys = list(zipped[1])
        for r in range(num_rings):
            cur_xs = xs[r*N_samples:(r+1)*N_samples]
            cur_ys = ys[r*N_samples:(r+1)*N_samples]
            plt.fill(cur_xs, cur_ys, color=favblue, alpha=0.2)
            #plt.plot(xs, ys, color=favblue, alpha=0.5)#, label='{0} iterations'.format(t))
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)

        plt.savefig("figures/dists_{}.png".format(i))
        plt.savefig("figures/dists_{}.pdf".format(i), pad_inches=0.05, bbox_inches='tight')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
