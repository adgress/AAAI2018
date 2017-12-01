import numpy as np
import cvxpy as cvx
from numpy import random
from numpy.linalg import norm


def _solve_mixed_guidance(X, Y, lambda_l2, lambda_mixed, E):
    n, p = X.shape
    w = cvx.Variable(p)
    b = cvx.Variable(1)
    z = cvx.Variable(E.shape[0])
    v = X * w + b - Y
    loss = (1.0 / n) * cvx.sum_squares(v)
    reg_l2 = lambda_l2 * cvx.sum_squares(w)
    reg_mixed = lambda_mixed * cvx.sum_entries(z)
    objective = cvx.Minimize(loss + reg_l2 + reg_mixed)
    constraints = [z >= 0, E*w + z >= 0]
    prob = cvx.Problem(objective, constraints)
    try:
        prob.solve(solver='SCS')
        assert w.value is not None
        w_value = np.squeeze(np.asarray(w.value))
        b_value = b.value
    except:
        print('CVX failed')
        w_value = np.zeros(p)
        b_value = 0
    return w_value, b_value


def solve_sign_feature_guidance(X, Y, lambda_l2, lambda_mixed, signs):
    p = X.shape[1]
    assert signs.size == p
    E = np.zeros((p, p))
    for idx, s in enumerate(signs):
        E[idx, idx] = s
    return _solve_mixed_guidance(X, Y, lambda_l2, lambda_mixed, E)


def solve_relative_feature_guidance(X, Y, lambda_l2, lambda_mixed, pairs):
    p = X.shape[1]
    num_pairs = len(pairs)
    E = np.zeros((num_pairs, p))
    for idx, (i, j) in enumerate(pairs):
        E[idx, i] = 1
        E[idx, j] = -1
    return _solve_mixed_guidance(X, Y, lambda_l2, lambda_mixed, E)



if __name__ == '__main__':
    n = 5
    p = 10
    X = random.rand(n, p)
    w = random.randn(p)
    b = random.randn(1)
    Y = X.dot(w) + b + random.normal(scale=.05, size=n)
    #w_hat, b_hat = solve_sign_guidance(X, Y, .001, 1, np.sign(w))
    w_hat, b_hat = solve_relative_feature_guidance(X, Y, .001, 1, [(0, 1)])
    print('')
