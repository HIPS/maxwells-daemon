# A simple experiment to show that the variational distribution implied by SGD abd Bayes-SGD
# adapts to the true distribution.

import numpy as np
import pickle
from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import sgd, entropic_descent, adaptive_entropic_descent, adaptive_sgd

# ------ Problem parameters -------
D = 2
def rosenbrock(w):
    x = w[1]
    y = w[0]
    return (100.0*(x-y**2.0)**2.0 + (1-y)**1.0)/1000
nllfun = lambda x: np.log(rosenbrock(x)+0.1)
nllfunt = lambda x, t : rosenbrock(x)

# ------ Variational parameters -------
seed = 0
x_init_scale = 2.0
v_init_scale = 2.0
alpha = 0.01
decay = 0.9
theta = 0.1
N_iter = 50
meta_alpha = 0.0001
meta_decay = 0.0001


# ------ Plot parameters -------
N_samples = 100

def run():
    print "Running experiment..."
    sgd_optimized_points = []
    for i in xrange(N_samples):
        rs = RandomState((seed, i))
        x0 = rs.randn(D) * x_init_scale
        v0 = rs.randn(D) * v_init_scale
        sgd_optimized_points.append(
            sgd(grad(nllfunt), x=x0, v=v0, learn_rate=alpha, decay=decay, iters=N_iter))

    return sgd_optimized_points, entropy

def plot():
    print "Plotting results..."
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
          sgd_optimized_points, ed_optimized_points, aed_optimized_points, asgd_optimized_points, entropy = pickle.load(f)

    xlims = [-3.0, 3.0]
    ylims = [-3.0, 6.0]

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    x = np.linspace(*xlims, num=100)
    y = np.linspace(*ylims, num=100)
    X, Y = np.meshgrid(x, y)
    zs = np.array([nllfun(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig("nll_surface.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    CS = plt.contour(X, Y, np.exp(-Z))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig("true_density.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    CS = plt.contour(X, Y, np.exp(-Z))
    plt.clabel(CS, inline=1, fontsize=10)
    for point in sgd_optimized_points:
        ax.plot(point[0], point[1], 'x')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    plt.savefig("true_density_with_sgd_points.png")


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
