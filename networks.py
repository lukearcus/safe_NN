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
        pred_deriv = torch.sum(torch.multiply(nn_grad, deriv), 1)
        #pred_deriv = nn_grad@deriv.T
        return pred_deriv
