"""A figure to show the nonparametric-ness of the implicit distributions."""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
from collections import defaultdict

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import sgd_damped

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
init_scale = 2.5
init_mu = np.array([-1.0, 1.0])
N_iter = 300
alpha = 0.01

grad_threshold=0.25

# ------ Plot parameters -------
N_samples = 250
N_samples_trails = 0
sample_trails_ix = [int(i) for i in np.floor(np.linspace(0,N_samples, N_samples_trails))][:-1]
N_snapshots = 2
spacing = N_iter / N_snapshots
snapshot_times = range(0,N_iter+1,spacing)
trail_lengths = N_iter
kernel_width = 0.1

def make_circle(N=100):
    th = np.linspace(0, 2 * np.pi, N)
    return np.concatenate((np.cos(th)[None, :], np.sin(th)[None, :]), axis=0).T

def run():
    all_xs = make_circle(N_samples) * init_scale + init_mu

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        def callback(x, t, g, v):
            if i in sample_trails_ix:
                results[("trail_x", i)].append(x.copy())
            results[("likelihood", i)].append(-nllfun(x))
            if t in snapshot_times:
                results[("all_x", t)].append(x.copy())

        x = sgd_damped(gradfun, x=all_xs[i], iters=N_iter, learn_rate=alpha,
                       decay=0.0, callback=callback, width=grad_threshold)
        callback(x, N_iter, 0.0, 0.0)
    return results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    with open('results.pkl') as f:
          results = pickle.load(f)

    # ------ contour plots --------
    rc('font',**{'family':'serif'})

    xlimits = [-4, 4]
    ylimits = [-4.1, 4]

    tp_levels = [0.2,] # 1.0, 10.0]   # Contour levels of true posterior
    contour_colours = ['red', 'limegreen', 'blue']

    x = np.linspace(*xlimits, num=200)
    y = np.linspace(*ylimits, num=200)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    zs = np.array([nllfun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    CS = plt.contour(X, Y, np.exp(-Z), tp_levels, colors='k')#, ls='--', label='True posterior')

    # Plot trails underneath.
    for i in sample_trails_ix:
        zipped = zip(*results[("trail_x", i)])
        xs = list(zipped[0])
        ys = list(zipped[1])
        plt.plot(xs, ys, 'b', alpha=0.5)

    for i, t in enumerate(snapshot_times):
        print "Plotting contour at time {0}".format(t)
        zipped = zip(*results[("all_x", t)])
        xs = list(zipped[0])
        ys = list(zipped[1])
        plt.fill(xs, ys, color=contour_colours[i], alpha=0.2 * (i+1))
        plt.plot(xs, ys, color=contour_colours[i], alpha=1.0)#, label='{0} iterations'.format(t))

    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    proxy = [plt.Rectangle((0,0),1,1,fc=c, edgecolor=c) for i, c in enumerate(contour_colours)]
    names = ['{0} iterations'.format(t) for t in snapshot_times]
    proxy.insert(0, plt.Line2D([],[],color='black'))
    names.insert(0, 'True posterior')

    plt.legend(proxy, names, loc=0, frameon=False, prop={'size':'10'})
    ax.set_yticks([])
    ax.set_xticks([])
    fig.set_size_inches((5,4))
    plt.savefig("dists.png")
    plt.savefig("dists.pdf".format(t), pad_inches=0.05, bbox_inches='tight')

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[("likelihood", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("likelihood", i)][t] for i in xrange(N_samples)])
              for t in xrange(N_iter)])
    plt.savefig("likelihoods.png")


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
