import numpy as np
import numpy.random as npr
from numpy.linalg import norm

def sgd(grad, x, v=None, callback=None, iters=200, learn_rate=0.1, decay=0.9):
    """Stochastic gradient descent with momentum."""
    if v is None: v = np.zeros(len(x))
    for t in xrange(iters):
        g = grad(x, t)
        if callback: callback(x, t, g, v)
        v = v - g
        x += learn_rate * v
        v *= decay
    return x

def entropic_descent(grad, x, v=None, callback=None, iters=200, learn_rate=0.1, decay=0.9, theta=0.1, rs=None):
    """Stochastic gradient descent with momentum, as well as velocity rotation.
    decay controls the amount velocities shrink each iteration.
    theta controls the amount velocities rotate each iteration."""
    if v is None: v = np.zeros(len(x))
    if rs is None: rs = npr.RandomState(0)
    for t in xrange(iters):
        g = grad(x, t)
        if callback: callback(x, t, g, v)
        v = v - g
        x += learn_rate * v
        r = rs.randn(len(v))
        v = v * np.cos(theta) + np.sin(theta) * r * norm(v) / norm(r)  # This is not quite correct yet.
        v *= decay
    return x


def adaptive_entropic_descent(grad, x, v=None, callback=None, iters=200,
                              init_learn_rate=0.1, init_log_decay=np.log(0.9),
                              meta_learn_rate=0.01, meta_decay=0.01):
    """Stochastic gradient descent with momentum with auto-adapting learning rate and decay.
    Hyperparameters are updated using gradient descent with stepsizes given by the meta params."""
    if v is None: v = np.zeros(len(x))
    learn_rates = np.full(len(x), init_learn_rate)
    log_decays = np.full(len(x), init_log_decay)
    for t in xrange(iters):
        g = grad(x, t)
        if callback: callback(x, t, g, v)
        v = v * np.exp(log_decays) - g
        log_decays += meta_decay * (learn_rates * ( v + g ) + 1)
        x += learn_rates * v
        learn_rates += meta_learn_rate * g * v
    return x


def adaptive_sgd(grad, x, v=None, callback=None, iters=200,
                              init_learn_rate=0.1, init_log_decay=np.log(0.9),
                              meta_learn_rate=0.01, meta_decay=0.01):
    """Stochastic gradient descent with momentum with auto-adapting learning rate and decay.
    Hyperparameters are updated using gradient descent with stepsizes given by the meta params."""
    if v is None: v = np.zeros(len(x))
    learn_rates = np.full(len(x), init_learn_rate)
    log_decays = np.full(len(x), init_log_decay)
    for t in xrange(iters):
        g = grad(x, t)
        if callback: callback(x, t, g, v)
        v = v * np.exp(log_decays) - g
        log_decays += meta_decay * (learn_rates * ( v + g ))
        x += learn_rates * v
        learn_rates += meta_learn_rate * g * v
    return x

def entropic_descent2(grad, x_scale, callback=None, epsilon=0.1,
                      gamma=0.1, alpha=0.1, annealing_schedule=None, rs=None):
    """Stochastic gradient descent with momentum and velocity randomization.
       gamma controls the amount velocities randomize each iteration.
       epsilon is roughly the scale of the square root of
       the largest eigenvalue of the Hessian of the log-posterior at the mode.
       rs is a RandomState.
       alpha is the integration step-size."""
    D = len(x_scale)
    x = rs.randn(D) * x_scale
    v = rs.randn(D)
    entropy = 0.5 * D * (1 + np.log(2*np.pi)) + np.sum(np.log(x_scale)) + 0.5 * (D - norm(v) **2)
    for t, anneal in enumerate(annealing_schedule):
        if callback: callback(x, t, v, entropy)
        entropy += 0.5 * norm(v) ** 2
        neg_dlog_init = x / x_scale**2
        g = anneal * grad(x, t) + (1 - anneal) * neg_dlog_init
        e = anneal * epsilon    + (1 - anneal) * x_scale
        v -= e * alpha * g
        x += e * alpha * v
        entropy -= 0.5 * norm(v) ** 2
        v = v * np.sqrt(1-gamma**2) + rs.randn(len(v)) * gamma
    if callback: callback(x, t + 1, v, entropy)
    return x, entropy

def entropic_descent_deterministic(grad, x_scale, callback=None, epsilon=0.1,
                                   gamma=0.1, alpha=0.1, annealing_schedule=None,
                                   rs=None, scale_calc_method='gradient',
                                   hessian=None):
    """Changes scale at each annealing step by estimating the change in curvature."""
    if scale_calc_method == 'gradient':
        def calc_scale(cur_anneal, prev_anneal):
            return np.sqrt(np.abs(convex_comb(prev_anneal, neg_dlog_final, neg_dlog_init) /
                                  convex_comb(cur_anneal,  neg_dlog_final, neg_dlog_init)))
    elif scale_calc_method == 'exact_hessian':
        # THIS ONLY WORKS FOR A GUASSIAN. JUST TESTING.
        def calc_scale(cur_anneal, prev_anneal):
            hess_final = hessian
            hess_init = 0.5 * 1 /  x_scale**2
            return np.full(D, np.sqrt(convex_comb(prev_anneal, hess_final, hess_init) /
                                      convex_comb(cur_anneal,  hess_final, hess_init)))
    else:
        raise Exception("{0} not valid".format(scale_calc_method))

    def convex_comb(f, A, B):
        return f * A + (1 - f) * B

    D = len(x_scale)
    x = rs.randn(D) * x_scale
    v = rs.randn(D)
    entropy = 0.5 * D * (1 + np.log(2*np.pi)) + np.sum(np.log(x_scale))
    prev_anneal = 0.0
    for t, anneal in enumerate(annealing_schedule):
        if callback: callback(x, t, v, entropy)
        neg_dlog_init = x / x_scale**2
        neg_dlog_final = grad(x, t)
        g = convex_comb(anneal, neg_dlog_final, neg_dlog_init)
        e = convex_comb(anneal, epsilon, x_scale)
        v -= e * alpha * g
        x += e * alpha * v

        scale_change = calc_scale(anneal, prev_anneal)
        v = v * scale_change
        entropy += np.sum(np.log(scale_change))
        prev_anneal = anneal

    if callback: callback(x, t + 1, v, entropy)
    return x, entropy
