import numpy as np
import torch
from torch import nn

def train_lyap(data, model, device):
    size = len(data)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for batch, trajectory in enumerate(data):
        states = np.vstack(trajectory[0])
        derivs = np.vstack(trajectory[1])
        state, deriv = torch.from_numpy(states.T), torch.from_numpy(np.array(derivs))
        state, deriv = state.to(device, dtype=torch.float32), deriv.to(device, dtype=torch.float32)
        zeros = torch.zeros_like(state)
        zero_val = model(zeros)
        tau = 1 
        #state.requires_grad = True

        # Compute prediction error
        pred = model(state)

        pred_deriv = model.get_deriv(state,deriv) 
        
        softplus = nn.Softplus()
        loss = torch.max(torch.max((-pred+zero_val),(pred_deriv-tau)))
        # Backpropagation
        loss.backward()
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #if batch%5 == 4:
            #print("batch {} of {} completed".format(batch+1, size))
