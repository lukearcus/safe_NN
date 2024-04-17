import numpy as np
import torch
from torch import nn

class template_NN(nn.Module):

    def forward(self, x):
        #x = self.flatten(x)
        output = self.forward_stack(x)
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

class test_NN(template_NN):
    def __init__(self):
        super().__init__()
        
    

        self.flatten = nn.Flatten()
        self.forward_stack = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            #nn.ReLU()
        )
        #def init_weights(m):
        #    if isinstance(m, nn.Linear):
        #        #torch.nn.init.xavier_uniform(m.weight)
        #        m.weight.data.fill_(0.01)
        #        m.bias.data.fill_(0.01)
        #self.linear_tanh_stack.apply(init_weights)


class structural_lyapunov(template_NN):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.forward_stack = nn.Sequential(
            nn.Linear(2, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 8, bias=False), # might nit be big enough?
            nn.Tanh(),
            nn.Linear(8, 8, bias=False), # might nit be big enough?
            nn.Tanh(),
            nn.Linear(8, 1, bias=False),
            nn.ReLU() # Try quadratic output layer?
        )
