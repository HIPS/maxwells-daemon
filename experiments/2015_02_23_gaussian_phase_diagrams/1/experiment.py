import numpy as np
import matplotlib.pyplot as plt

def run():
    epsilon = 0.05
    x_scale = 1.0
    x_init = -3.8
    decay_rate = 0.3
    N_steps = 100
    lim = 5.0
    thin = 10
    # updater = gaussian_hmc_update
    updater = gaussian_aed_update

    mu = np.array([x_init, 0.0])
    inv_chol = np.array([[x_scale, 0.0],[0.0, 1.0]])
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    ax.plot([-lim, lim], [0,0], color="black")
    ax.plot([0,0], [-lim, lim], color="black")
    for i in range(N_steps):
        if i % thin == 0:
            plot_gaussian(ax, mu, inv_chol)
        mu, inv_chol = updater(mu, inv_chol, epsilon, decay_rate)
    plot_gaussian(ax, np.zeros(2), np.eye(2), color="black", ls='--')
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    plt.savefig('phase_diagram.png')

def plot_gaussian(ax, mu, inv_chol, **kwargs):
    th = np.linspace(0, 2 * np.pi, 100)
    V = np.concatenate((np.cos(th)[None, :], np.sin(th)[None, :]), axis=0)
    V = inv_chol.dot(V) + mu[:, None]
    x = V[0, :]
    y = V[1, :]
    ax.plot(x, y, **kwargs)

def gaussian_aed_update(mu, inv_chol, epsilon, decay_rate):
    H_1 = np.array([[1.0, 0.0],[- epsilon / 2, 1.0]]) # transformation of v (epsilon / 2 size)
    H_2 = np.array([[1.0, epsilon],[0.0, 1.0]]) # transformation of x (epsilonsize)
    H_3 = np.array([[1.0, 0.0],[0.0, 1.0 - decay_rate * epsilon]])
    for op_mat in [H_1, H_2, H_1, H_3]:
        inv_chol = op_mat.dot(inv_chol)
        mu = op_mat.dot(mu)
    return mu, inv_chol

def gaussian_hmc_update(mu, inv_chol, epsilon, decay_rate):
    H_1 = np.array([[1.0, 0.0],[- epsilon / 2, 1.0]]) # transformation of v (epsilon / 2 size)
    H_2 = np.array([[1.0, epsilon],[0.0, 1.0]]) # transformation of x (epsilonsize)
    H_3 = np.array([[1.0, 0.0],[0.0, np.sqrt(1 - (decay_rate * epsilon)**2 )]])
    for op_mat in [H_1, H_2, H_1, H_3]:
        inv_chol = op_mat.dot(inv_chol)
        mu = op_mat.dot(mu)

    cov = inv_chol.dot(inv_chol.T)
    cov[1,1] += (decay_rate * epsilon)**2
    inv_chol = np.linalg.cholesky(cov)
    return mu, inv_chol

if __name__ == "__main__":
    run()
