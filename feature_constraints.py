import numpy as np
import cvxpy as cvx
from numpy import random
from numpy.linalg import norm
from scipy import optimize

def unpack_same_sign_hinge_primal(v, opt_data):
    p = opt_data['p']
    w = v[:p]
    b = v[p]
    z = v[p+1:]
    return w, b, z

def eval_same_sign_hinge_primal(v, opt_data):
    n = opt_data['n']
    x = opt_data['x']
    y = opt_data['y']
    pairs = opt_data['pairs']
    lambda_l2 = opt_data['lambda_l2']
    lambda_slack = opt_data['lambda_slack']
    lambda_features = opt_data['lambda_features']

    w, b, z = unpack_same_sign_hinge_primal(v, opt_data)
    loss = (1.0/n)*norm(x.dot(w) + b - y)**2
    reg1 = lambda_l2*norm(w)**2
    reg2 = lambda_slack*z.sum()
    loss_guidance = 0
    for idx, (i, j, s) in enumerate(pairs):
        l = s*w[i]*w[j]
        l += z[idx]
        loss_guidance += max(-l, 0)
    loss_guidance *= lambda_features
    return loss + loss_guidance + reg1 + reg2


def solve_pairwise_sign_feature_guidance(X, Y, lambda_l2, lambda_slack, lambda_features, pairs):
    n, p = X.shape
    v0 = np.zeros(p + 1 + len(pairs))
    bounds = [(None, None)] * p + [(None, None)] + [(0, None)] * len(pairs)
    opt_data = {
        'n': n,
        'p': p,
        'x': X,
        'y': Y,
        'lambda_l2': lambda_l2,
        'lambda_slack': lambda_slack,
        'lambda_features': lambda_features,
        'pairs': pairs
    }
    results = optimize.minimize(
        lambda v: eval_same_sign_hinge_primal(v, opt_data),
        v0,
        method='SLSQP',
        jac=None,
        options=None,
        bounds=bounds,
        constraints=[]
    )
    w, b, z = unpack_same_sign_hinge_primal(results.x, opt_data)
    return w, b


def _solve_feature_guidance_cvx(X, Y, lambda_l2, lambda_mixed, E):
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
    assert len(signs) == p
    E = np.zeros((p, p))
    for idx, s in enumerate(signs):
        E[idx, idx] = s
    return _solve_feature_guidance_cvx(X, Y, lambda_l2, lambda_mixed, E)


def solve_relative_feature_guidance(X, Y, lambda_l2, lambda_mixed, pairs):
    p = X.shape[1]
    num_pairs = len(pairs)
    E = np.zeros((num_pairs, p))
    for idx, (i, j) in enumerate(pairs):
        E[idx, i] = 1
        E[idx, j] = -1
    return _solve_feature_guidance_cvx(X, Y, lambda_l2, lambda_mixed, E)


if __name__ == '__main__':
    n = 5
    p = 10
    X = random.rand(n, p)
    w = random.randn(p)
    b = random.randn(1)
    Y = X.dot(w) + b + random.normal(scale=.05, size=n)
    signs = np.sign(w).tolist()
    w_hat_sign, b_hat_sign = solve_sign_feature_guidance(X, Y, .001, 1, signs)
    w_hat_relative, b_hat_relative = solve_relative_feature_guidance(X, Y, .001, 1, [(0, 1)])
    sign_pairs = [(0, 1, signs[0]*signs[1])]
    w_hat_pairwise_sign, b_hat_pairwise_sign = solve_pairwise_sign_feature_guidance(X, Y, .001, .001, .001, sign_pairs)
