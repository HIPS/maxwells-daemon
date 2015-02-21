"""First real experiment - how well do we do on MNIST?"""

import numpy as np
from numpy.linalg import norm
import pickle
from collections import defaultdict

from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.optimizers import entropic_descent2
from maxwell_d.nn_utils import make_nn_funs
from maxwell_d.data import load_data_subset


# ------ Problem parameters -------
layer_sizes = [784, 200, 10]
batch_size = 200
N_train = 10**3
N_tests = 10**3

# ------ Variational parameters -------
seed = 0
init_scale = 1.0
epsilon = 0.1
gamma = 0.1
N_iter = 1000
alpha = 0.1
annealing_schedule = np.linspace(0, 1, N_iter)

# ------ Plot parameters -------
N_samples = 3
N_checkpoints = 10
thin = np.ceil(N_iter/N_checkpoints)

def run():
    (train_images, train_labels),\
    (tests_images, tests_labels) = load_data_subset(N_train, N_tests)
    parser, pred_fun, nllfun, frac_err = make_nn_funs(layer_sizes)

    print "Running experiment..."
    results = defaultdict(list)
    for i in xrange(N_samples):
        x_init_scale = np.full(len(parser.vect), init_scale)

        def indexed_loss_fun(w, i_iter):
            rs = RandomState((seed, i, i_iter))
            idxs = rs.randint(N_train, size=batch_size)
            return nllfun(w, train_images[idxs], train_labels[idxs])
        gradfun = grad(indexed_loss_fun)

        def callback(x, t, v, entropy):
            results[("entropy", i)].append(entropy)
            results[("v_norm", i)].append(norm(v))
            results[("minibatch_likelihood", i)].append(-indexed_loss_fun(x, t))

            if t % thin == 0 or t == N_iter or t == 0:
                results[('iterations', i)].append(t)
                results[("train_likelihood", i)].append(-nllfun(x, train_images, train_labels))
                results[("tests_likelihood", i)].append(-nllfun(x, tests_images, tests_labels))
                results[("tests_error", i)].append(frac_err(x, tests_images, tests_labels))
                print "Iteration {0:5i} Train likelihood {1:2.4f}  Test likelihood {2:2.4f}" \
                      "  Test Err {3:2.4f}".format(t, results[("train_likelihood", i)][-1],
                                                      results[("tests_likelihood", i)][-1],
                                                      results[("tests_error",      i)][-1])

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
        plt.plot(results[("v_norm", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("v_norm", i)][t] for i in xrange(N_samples)])
              for t in xrange(N_iter)])
    plt.savefig("v_norms.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[("minibatch_likelihood", i)])
    ax = fig.add_subplot(212)
    plt.plot([np.mean([results[("minibatch_likelihood", i)][t] for i in xrange(N_samples)]) for t in xrange(N_iter)])
    plt.savefig("minibatch_likelihoods.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[('iterations', i)],
                 [estimate_marginal_likelihood(results[("train_likelihood", i)][t_ix],
                                               results[("entropy", i)][t])
                  for t_ix, t in enumerate(results[('iterations', i)])])
    ax = fig.add_subplot(212)
    plt.plot(results[('iterations', i)],
             [np.mean([estimate_marginal_likelihood(results[("train_likelihood", i)][t_ix],
                                                    results[("entropy", i)][t])
                       for i in xrange(N_samples)])
              for t_ix, t in enumerate(results[('iterations', 0)])])
    plt.savefig("marginal_likelihoods.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[('iterations', i)],
                 [results[("tests_likelihood", i)][t] for t in xrange(len(results[('iterations', i)]))],)
    ax = fig.add_subplot(212)
    plt.plot(results[('iterations', i)],
             [np.mean([results[("tests_likelihood", i)][t] for i in xrange(N_samples)])
              for t in xrange(len(results[('iterations', 0)]))])
    plt.savefig("test_likelihoods.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(211)
    for i in xrange(N_samples):
        plt.plot(results[('iterations', i)],
                 [results[("tests_error", i)][t] for t in xrange(len(results[('iterations', i)]))])
    ax = fig.add_subplot(212)
    plt.plot(results[('iterations', 0)],
             [np.mean([results[("tests_error", i)][t] for i in xrange(N_samples)])
              for t in xrange(len(results[('iterations', 0)]))])
    plt.savefig("test_errors.png")

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
