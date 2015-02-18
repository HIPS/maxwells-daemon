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