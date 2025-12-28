import numpy as np
import torch 

def grad_F(u, ds):
    return ds.alpha * ds.N * u - ds.x_hat

# To save memory, only update within rng is performed
def inner_gd(u_last, v_last, ds, params, rng):
    # copy u N times to initialize the N concurrent subproblems
    u = u_last.unsqueeze(0).repeat(len(rng), 1, 1) 
    for i in range(params.n_iter):
        # X_j is essentially x_j repeated K times, altogether packed as (N, K, d)
        Xj = ds.X[rng, :].unsqueeze(1).expand(-1, ds.K, -1) 
        _XjU = torch.einsum('nkd, nkd -> nk', Xj, u)
        _grad_LSE = Xj * _XjU.softmax(dim = 1).unsqueeze(-1) # (N, K, d), (N, K) -> (N, K, d)
        grad_H = grad_F(u, ds) + v_last[rng, :] - grad_F(u_last, ds) + _grad_LSE
        u -= params.lr * grad_H 
    return u

def parallel_douglas_rachford(tau, inits, n_iter, inner_params, ds):
    '''
        ds: (dataset)
            N: number of samples
            K: number of classes
            d: dimension of feature (dimension of x plus 1)
            alpha: regularization
            X(N, d): features
            Y(N, 1): true labels
            x_hat(K, d): accumulated features by class
        inner_params:
            n_iter: step of inner gradient descent
            lr: step size
        inits:
            u0(K, d): initial linear weights
    '''
    u_t = inits.u0
    v_t = inits.v0 
    batch_size = 1000
    for t in range(n_iter):
        # solve N argmins in parallel in batches
        for start in range(0, ds.N, batch_size):
            rng = range(start, min(start + batch_size, ds.N))
            u_batch = inner_gd(u_t, v_t, ds, inner_params, rng)
            # update v
            v_t[rng, :] += tau * (grad_F(u_batch, ds) - grad_F(u_t, ds))
        # update u
        u_t = 1 / (ds.alpha * ds.N) * (ds.x_hat + v_t.sum(dim = 0))
    return u_t 
        