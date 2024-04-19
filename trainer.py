import numpy as np
import torch
from torch import nn

def train_lyap(data, model, device, tau):
    size = len(data)
    model.train()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    max_val = None
    total_loss = 0
    num_violations = 0
    #tau = 1e-10
    for batch, trajectory in enumerate(data):
        states = np.vstack(trajectory[0])
        derivs = np.vstack(trajectory[1])
        state, deriv = torch.from_numpy(states.T), torch.from_numpy(np.array(derivs))
        state, deriv = state.to(device, dtype=torch.float32), deriv.to(device, dtype=torch.float32)
        zeros = torch.zeros_like(state)
        zero_val = model(zeros)
        #state.requires_grad = True

        # Compute prediction error
        pred = model(state)

        pred_deriv = model.get_deriv(state,deriv) 
        #if max_val is not None:
        #    #max_val = torch.max(max_val, torch.max(pred_deriv))
        #    max_val = torch.max(max_val,torch.max(torch.max((-pred+zero_val),(pred_deriv-tau))))
        #else:
        #    #max_val = torch.max(pred_deriv)
        #    max_val = torch.max(torch.max((-pred+zero_val),(pred_deriv-tau)))
        #
        #loss = max_val
        # loss above is max_i, loss below is sum_i
        
        eta_i = torch.max(torch.max(-pred+zero_val),torch.max(pred_deriv))+tau
        if eta_i >= 0:
            total_loss += eta_i-tau
            if eta_i-tau >= 0:
                num_violations += 1
        #relu = torch.nn.ReLU()

        #loss = relu(tau+)-tau
        #total_loss += loss
        
        # Backpropagation
    #print(pred-zero_val)
    #print(pred_deriv)
    if type(total_loss) is not int:
        total_loss.backward()
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss, num_violations
    #if batch%5 == 4:
        #print("batch {} of {} completed".format(batch+1, size))

def train_disc_lyap(data, model, device):
    size = len(data)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
    max_val = None
    for batch, trajectory in enumerate(data):
        states = np.vstack(trajectory[0])
        next_states = np.vstack(trajectory[1])
        state, next_s = torch.from_numpy(states), torch.from_numpy(np.array(next_states))
        state, next_s = state.to(device, dtype=torch.float32), next_s.to(device, dtype=torch.float32)
        zeros = torch.zeros_like(state)
        zero_val = model(zeros)
        tau = 0 
        #state.requires_grad = True

        # Compute prediction error
        pred = model(state)
        pred_next = model(next_s)

        #pred_deriv = model.get_deriv(state,deriv) 
        pred_deriv = pred_next-pred

        if max_val is not None:
            max_val = torch.max(max_val,torch.max(torch.max((-pred+zero_val),(pred_deriv-tau))))
        else:
            max_val = torch.max(torch.max((-pred+zero_val),(pred_deriv-tau)))
        
    loss = max_val
    #loss = torch.max(torch.max((-pred+zero_val),(pred_deriv-tau)))
    # Backpropagation
    loss.backward()
    #loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #if batch%5 == 4:
        #print("batch {} of {} completed".format(batch+1, size))
