import numpy as np
import pickle
from funkyyak import grad
from canonical_data import load_data, random_partition
from util import RandomState

# Network params
layer_sizes = [784, 10]
N_layers = len(layer_sizes) - 1
# Training params
N_batch = 200
N_iters = 50
alpha = 1.0
beta = 0.99
theta = 0.1
seed = 0
# Data params
N_train = 1000
N_valid = 1000
N_total = N_train + N_valid

def run():
    rs = RandomState((seed, "top_rs"))
    all_data, _ = load_data("mnist", N=N_total, normalize=True)
    train_data, valid_data = random_partition(all_data, rs, [N_train, N_valid])
    w_parser, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes)
    N_weights = w_parser.vect.size
    loss_grad = grad(loss_fun)
    W = rs.randn(N_weights)
    V = rs.randn(N_weights)
    train_loss, valid_loss = np.zeros(N_iters), np.zeros(N_iters)
    for i in range(N_iters):
        cur_batch = random_partition(train_data, rs, [N_batch])
        V -= loss_grad(W, **cur_batch)
        W += alpha * V
        V = beta * random_rotation(V, theta, rs)
        train_loss[i] = loss_fun(W, **train_data)
        valid_loss[i] = loss_fun(W, **valid_data)

    L_var = train_loss - np.arange(N_iters) * N_weights * np.log(beta)
    return train_loss, valid_loss, L_var

def random_rotation(V, theta, rs):
    R = rs.randn(V.size)
    R = R / np.linalg.norm(R)
    R = R * np.linalg.norm(V) - np.dot(R, V)
    assert np.allclose(np.dot(R, V), 0.0)
    assert np.allclose(np.linalg.norm(V), np.linalg.norm(R))
    return np.cos(theta) * V + np.sin(theta) * R

def plot():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    with open('results.pkl') as f:
        train_loss, valid_loss, L_var = pickle.load(f)

    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(-train_loss, label='Train log prob')
    ax.plot(-valid_loss, label='Valid log prob')
    ax.plot(L_var,       label='Variational lower bound')
    ax.set_ylabel('Log prob per datum')
    ax.set_xlabel('Iteration')
    ax.set_title('Learning curves')
    plt.savefig("learning_curve.png")

if __name__ == "__main__":
    results = run()
    with open('results.pkl', 'w') as f:
        pickle.dump(results, f, 1)
    plot()
