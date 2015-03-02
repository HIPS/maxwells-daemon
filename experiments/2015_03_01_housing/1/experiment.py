"""Experiment on a small real dataset."""
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import pickle
from collections import defaultdict
from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import sgd_entropic
from maxwell_d.nn_utils import make_regression_nn_funs
from maxwell_d.data import load_boston_housing

# ------ Problem parameters -------
layer_sizes = [13, 100, 1]
train_frac = 0.1
# ------ Variational parameters -------
seed = 0
init_scale = 0.1
N_iter = 500
alpha_un = 0.004
reg = 0.1
# ------ Plot parameters -------
N_samples = 1
N_checkpoints = 50
thin = np.ceil(N_iter/N_checkpoints)

def neg_log_prior(w):
    # return 0.5 * np.dot(w, w) / init_scale**2
    return 0.5 * reg * np.dot(w, w)
    # return 0.0

def run():
    train_inputs, train_targets,\
    tests_inputs, tests_targets, unscale_y = load_boston_housing(train_frac)
    N_train = train_inputs.shape[0]
    batch_size = N_train
    alpha = alpha_un / N_train
    parser, pred_fun, nllfun, rmse = make_regression_nn_funs(layer_sizes)
    N_param = len(parser.vect)

    def indexed_loss_fun(w, i_iter):
        rs = RandomState((seed, i, i_iter))
        idxs = rs.randint(N_train, size=batch_size)
        nll = nllfun(w, train_inputs[idxs], train_targets[idxs]) * N_train
        nlp = neg_log_prior(w)
        return nll + nlp
    gradfun = grad(indexed_loss_fun)

    def callback(x, t, entropy):
        results["entropy_per_dpt"     ].append(entropy / N_train)
        results["x_rms"               ].append(np.sqrt(np.mean(x * x)))
        results["minibatch_likelihood"].append(-indexed_loss_fun(x, t))
        results["log_prior_per_dpt"   ].append(-neg_log_prior(x) / N_train)
        if t % thin != 0 and t != N_iter and t != 0: return
        results["iterations"      ].append(t)
        results["train_likelihood"].append(-nllfun(x, train_inputs, train_targets))
        results["tests_likelihood"].append(-nllfun(x, tests_inputs, tests_targets))
        results["train_rmse"      ].append(unscale_y(rmse(x, train_inputs, train_targets)))
        results["tests_rmse"      ].append(unscale_y(rmse(x, tests_inputs, tests_targets)))
        results["marg_likelihood" ].append(estimate_marginal_likelihood(
            results["train_likelihood"][-1], results["entropy_per_dpt"][-1]))
                                           
        print "Iteration {0:5} Train lik {1:2.4f}  Test lik {2:2.4f}" \
              "  Marg lik {3:2.4f}  Test RMSE {4:2.4f}".format(
                  t, results["train_likelihood"][-1],
                  results["tests_likelihood"][-1],
                  results["marg_likelihood" ][-1],
                  results["tests_rmse"      ][-1])

    all_results = []
    for i in xrange(N_samples):
        results = defaultdict(list)
        rs = RandomState((seed, i))
        sgd_entropic(gradfun, np.full(N_param, init_scale), N_iter, alpha, rs, callback)
        all_results.append(results)

    return all_results

def estimate_marginal_likelihood(likelihood, entropy):
    return likelihood + entropy

def plot():
    print "Plotting results..."
    with open('results.pkl') as f:
          results = pickle.load(f)

    first_results = results[0]
    # Diagnostic plots of everything for us.
    for key in first_results:
        plot_traces_and_mean(results, key)

    # Nice plots for paper.
    rc('font',**{'family':'serif'})
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    plt.plot(first_results["iterations"], first_results["train_rmse"], 'b', label="Train error")
    plt.plot(first_results["iterations"], first_results["tests_rmse"], 'g', label="Test error")
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    ax.set_ylabel('RMSE')

    ax = fig.add_subplot(212)
    plt.plot(first_results["iterations"], first_results["marg_likelihood"], 'r', label="Marginal likelihood")
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    ax.set_ylabel('Marginal likelihood')
    ax.set_xlabel('Training iteration')
    #low, high = ax.get_ylim()
    #ax.set_ylim([0, high])

    fig.set_size_inches((5,3.5))
    ax.legend(numpoints=1, loc=1, frameon=False, prop={'size':'12'})
    plt.savefig('marglik.pdf', pad_inches=0.05, bbox_inches='tight')

def plot_traces_and_mean(results, trace_type, X=None):
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    if X is None:
        X = np.arange(len(results[0][trace_type]))
    for i in xrange(N_samples):
        plt.plot(X, results[i][trace_type])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(trace_type)
    ax = fig.add_subplot(212)
    all_Y = [np.array(results[i][trace_type]) for i in range(N_samples)]
    plt.plot(X, sum(all_Y) / float(len(all_Y)))
    plt.savefig(trace_type + '.png')


if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
