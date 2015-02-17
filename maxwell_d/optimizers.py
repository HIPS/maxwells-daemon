import numpy as np
import numpy.random as npr
from numpy.linalg import norm

def sgd(grad, x, v=None, callback=None, iters=200, learn_rate=0.1, decay=0.9):
    """Stochastic gradient descent with momentum."""
    if v is None: v = np.zeros(len(x))
    for t in xrange(iters):
        g = grad(x, t)
        if callback: callback(x, t, g)
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
        if callback: callback(x, v, t, g)
        v = v - g
        x += learn_rate * v
        r = rs.randn(len(v))
        v = v * np.cos(theta) + np.sin(theta) * r * norm(v) / norm(r)  # This is not quite correct yet.
        v *= decay
    return x