import numpy as np
import torch
from torch import nn

class test_NN(nn.Module):
    def __init__(self):
        super().__init__()
        
    

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            #nn.ReLU()
        )

    def forward(self, x):
        #x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

    def get_deriv(self, x, deriv):
        S_clone = torch.clone(x).requires_grad_()
        nn_clone = self(S_clone)

        nn_grad = torch.autograd.grad(
            outputs=nn_clone,
            inputs=S_clone,
            grad_outputs=torch.ones_like(nn_clone),
            create_graph=True,
            retain_graph=True,
            # allow_unused=True,
        )[0]
        pred_deriv = nn_grad@deriv.T
        return pred_deriv

def train_lyap(data, model, device):
    size = len(data)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for batch, (state, deriv) in enumerate(data):
        state, deriv = torch.from_numpy(state), torch.from_numpy(np.array(deriv))
        state, deriv = state.to(device, dtype=torch.float32), deriv.to(device, dtype=torch.float32)

        tau = 1
        #state.requires_grad = True

        # Compute prediction error
        pred = model(state)
        pred_deriv = model.get_deriv(state,deriv) 
        
        softplus = nn.Softplus()
        loss = softplus(-pred-tau)+softplus(pred_deriv-tau)
        # Backpropagation
        loss.backward()
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()
