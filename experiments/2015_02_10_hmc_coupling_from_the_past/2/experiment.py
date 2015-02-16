import numpy as np
import pickle
from funkyyak import grad

from maxwell_d.util import RandomState
from maxwell_d.exact_rep import ExactRep

seed = 0
gradfun = grad(lambda x : 0.5 * np.dot(x, x))
xmax = 1.0
vmax = 1.0
all_N_iter = range(60)
N_iter_plot = 13
alpha = 0.5
beta = 0.75
N_back = 1000
N_back_keep = 50
precision = 16
total_bits = ((precision / 2) + 1) * 2
bits_per_iter = - np.log2(beta)

def valid_initialization(X, V):
    return np.all(np.abs(X.val) <= xmax) and \
           np.all(np.abs(V.val) <= vmax) and \
           V.aux.is_empty()

def initialize(rs):
    X = ExactRep(rs.uniform(-xmax, xmax, 1), nbits=precision)
    V = ExactRep(rs.uniform(-vmax, vmax, 1), nbits=precision)
    return X, V

def forward_iter(X, V):
    X.add(alpha * V.val)
    V.mul(beta).sub(gradfun(X.val))
    return X, V

def reverse_iter(X, V):
    V.add(gradfun(X.val)).div(beta)
    X.sub(alpha * V.val)
    return X, V

def random_unwind(X, V, rs, N_iter):
    X = X.copy()
    V = V.copy().randomize(rs)
    path = [X.val[0]]
    for i in range(N_iter):
        X, V = reverse_iter(X, V)
        path.append(X.val[0])
    is_valid = valid_initialization(X, V)
    return is_valid, path

def run():
    rs = RandomState((seed, "top"))
    X, V = initialize(rs)
    x0 = X.val
    forward_path = [X.val[0]]
    densities = []
    for i_iter in range(all_N_iter[-1] + 1):
        X, V = forward_iter(X, V)
        forward_path.append(X.val[0])
        if i_iter in all_N_iter:
            print i_iter
            paths = [random_unwind(X, V, rs, i_iter + 1) for i_back in range(N_back)]
            densities.append(np.mean([p[0] for p in paths]))
            if i_iter == N_iter_plot:
                backward_paths = paths[:N_back_keep]

    return forward_path, backward_paths, densities

def plot():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    with open('results.pkl') as f:
          forward_path, backward_paths, densities = pickle.load(f)

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    seen = {}
    for is_valid, path in backward_paths:
        if is_valid:
            color = "DarkOliveGreen"
            label = "" if "valid" in seen else "Valid backward pass"
            seen["valid"] = True
        else:
            color="DarkOrange"
            label = "" if "not_valid" in seen else "Failed backward pass"
            seen["not_valid"] = True

        ax.plot(path[::-1], color=color, label=label)
    ax.plot(forward_path, color="RoyalBlue", label="Forward pass")
    ax.set_ylim([-xmax, xmax])
    ax.set_ylabel('X')
    ax.set_xlabel('Iteration')
    ax.set_title('Optimization trajectory')
    ax.legend(loc=0, frameon=False)
    plt.savefig("optimization.png")

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    N_bits_nominal = (np.array(all_N_iter) + 1) * bits_per_iter
    N_bits_max = np.minimum(N_bits_nominal, total_bits)
    N_bits_actual = N_bits_nominal + np.log2(densities)
    ax.plot(N_bits_nominal, N_bits_max,    label="Theoretical max")
    ax.plot(N_bits_nominal, N_bits_actual, label="Estimated actual performance")
    ax.set_xlim([1, 20])
    ax.set_ylim([0, 20])
    ax.set_xlabel("Nominal bits")
    ax.set_ylabel("Actual bits")
    ax.set_title("Counting bits extracted during optimization")
    ax.legend(loc=0, frameon=False)
    plt.savefig("entropy_estimates.png")

if __name__ == '__main__':
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
