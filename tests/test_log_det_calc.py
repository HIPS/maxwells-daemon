import numpy as np

from maxwell_d.util import RandomState
from maxwell_d.optimizers import exact_log_det, approx_log_det

def test_exact_log_det():
    D = 100
    rs = RandomState(1)
    mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
    mvp = lambda v : np.dot(mat, v)
    assert exact_log_det(mvp, D) == np.log(np.linalg.det(mat))

def test_approx_log_det():
    D = 100
    rs = RandomState(0)
    mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
    mvp = lambda v : np.dot(mat, v)
    N_trials = 10000
    approx = 0
    for i in xrange(N_trials):
        approx += approx_log_det(mvp, D, rs)
    approx = approx / N_trials
    exact = exact_log_det(mvp, D)
    assert exact > approx > (exact - 0.1 * np.abs(exact))
    print exact, approx
