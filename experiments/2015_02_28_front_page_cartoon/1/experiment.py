"""An illustrative example of sgd as variational inference"""

import numpy as np
import pickle
from funkyyak import grad

from maxwell_d.util import RandomState

log_posterior = lambda x : - 0.5 * np.log(2 * np.pi) - 0.5 * x**2
init_samp = lambda rs : - 1.0 + 3.0 * rs.randn()
N_samp = 4 * 10**5
N_iters = 16
save_iters = [0, 5, 10, 15]
alpha = 0.11
x_max = 8.0
x_min = -8.0
N_bins = 100
bin_x = np.linspace(x_min, x_max, N_bins)
bin_width = (x_max - x_min) / (N_bins - 1)

def run():
    rs = RandomState(0)
    grad_posterior = grad(log_posterior)
    bin_counts = {i_iter : np.zeros(N_bins) for i_iter in save_iters}
    h = 1.0 / bin_width / N_samp
    for i_samp in xrange(N_samp):
        x = init_samp(rs)
        for i_iter in xrange(N_iters):
            if i_iter in bin_counts:
                bin_counts[i_iter][bin_idx(x)] += h
            x += alpha * grad_posterior(x)

    return bin_counts

def bin_idx(x):
    idx = int(np.round( (x - x_min) / bin_width ))
    return np.clip(idx, 0, N_bins - 1)

def plot():
    import matplotlib.pyplot as plt
    with open('results.pkl') as f:
          bin_counts = pickle.load(f)
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    posterior = np.exp(log_posterior(bin_x))
    for i_iter in save_iters:
        plt.plot(bin_x, bin_counts[i_iter], label='{0} iterations'.format(i_iter))
    plt.plot(bin_x, posterior, '--k', lw=3, label='True posterior')
    ax.legend(loc=1, frameon=False)    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0.0, 0.6)
    ax.set_xlim(x_min + 2*bin_width, x_max - 2*bin_width)
    plt.savefig('cartoon.png')

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
