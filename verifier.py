import torch
import numpy as np
from math import comb
from scipy.stats import beta as betaF
CONVERGE_TOL = 1e-4

def MC_test_lyap(num_samples, network, device, model):
    num_violations = 0
    converge_violations = 0
    for i in range(num_samples):
        times, states, derivs = model.return_trajectory(10)
        final_state = states.T[-1]
        if np.linalg.norm(final_state) > CONVERGE_TOL:
            converge_violations += 1
        states, derivs = np.vstack(states), np.vstack(derivs) 
        state, deriv = torch.from_numpy(states.T), torch.from_numpy(np.array(derivs))
        state, deriv = state.to(device, dtype=torch.float32), deriv.to(device, dtype=torch.float32)
        pred_V = network(state)
        pred_V_deriv = network.get_deriv(state,deriv) 
        
        if any(pred_V < 0) or any(pred_V_deriv > 0):
            num_violations += 1
    return num_violations/num_samples, converge_violations/num_samples

def verify_lyap(data, network, device, beta):
    size = len(data)
    num_violations = 0
    for batch, traj in enumerate(data):
        states = np.vstack(traj[0])
        derivs = np.vstack(traj[1])
        state, deriv = torch.from_numpy(states.T), torch.from_numpy(np.array(derivs))
        state, deriv = state.to(device, dtype=torch.float32), deriv.to(device, dtype=torch.float32)
        pred_0 = network(torch.zeros_like(state))
        pred_V = network(state)
        pred_V_deriv = network.get_deriv(state,deriv) 
        print(pred_V)
        print(pred_V_deriv)
        
        if any(pred_V - pred_0 <= 0) or any(pred_V_deriv > 0):
            num_violations += 1
        else:
            print(batch)
    eps = calc_eps_risk_complexity(beta, size, num_violations)
    return eps

def calc_eps_risk_complexity(beta, N, k):
    if k != 0:
        alphaL = betaF.ppf(beta, k, N-k+1)
        alphaU = 1-betaF.ppf(beta, N-k+1, k)
    else:
        alphaL = 0
        alphaU = 0

    m1 = np.expand_dims(np.arange(k, N+1),0)
    #m1[0,0] = k+1
    aux1 = np.sum(np.triu(np.log(np.ones([N-k+1,1])@m1),1),1)
    aux2 = np.sum(np.triu(np.log(np.ones([N-k+1,1])@(m1-k)),1),1)
    coeffs1 = np.expand_dims(aux2-aux1, 1)

    m2 = np.expand_dims(np.arange(N+1, 4*N+1),0)
    aux3 = np.sum(np.tril(np.log(np.ones([3*N,1])@m2)),1)
    aux4 = np.sum(np.tril(np.log(np.ones([3*N,1])@(m2-k))),1)
    coeffs2 = np.expand_dims(aux3-aux4, 1)

    def poly(t):
        val = 1
        val += beta/(2*N) 
        val -= (beta/(2*N))*np.sum(np.exp(coeffs1 - (N-m1.T)*np.log(t)))
        val -=(beta/(6*N))*np.sum(np.exp(coeffs2 + (m2.T-N)*np.log(t)))

        
        return val
    t1 = 1-alphaL
    t2 = 1
    poly1 = poly(t1)
    poly2 = poly(t2)

    if ((poly1*poly2)) > 0:
        epsL = 0
    else:
        while t2-t1 > 10**-10:
            t = (t1+t2)/2
            polyt  = poly(t)
            if polyt > 0:
                t1 = t
            else:
                t2 = t
        epsL = 1-t2

    t1 = 0
    t2 = 1-alphaU

    while t2-t1 > 10**-10:
        t = (t1+t2)/2
        polyt  = poly(t)
        if polyt > 0:
            t2 = t
        else:
            t1 = t
    epsU = 1-t1
    return epsU
