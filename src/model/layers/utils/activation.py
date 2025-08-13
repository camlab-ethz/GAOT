import torch
from torch import nn
import torch.nn.functional as F

class Activation(nn.Module):
    """
        Parameters:
        -----------
            x: torch.FloatTensor
                input tensor
        Returns:
        --------
            y: torch.FloatTensor
                output tensor, same shape as the input tensor, since it's element-wise operation
    """
    def __init__(self, activation:str):
        super().__init__()
        activation = activation.lower()
        if activation in ['sigmoid', 'tanh']:
            self.activation_fn = getattr(torch, activation)
        elif activation == "swish":
            self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
    
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = getattr(F, activation)
        self.activation = activation
    def forward(self, x):
        if self.activation == "swish":
            return x * torch.sigmoid(self.beta * x)
        elif self.activation == "gelu":
            return x * torch.sigmoid(1.702 * x)
        elif self.activation == "mish":
            return x * torch.tanh(F.softplus(x))
        else:
            return self.activation_fn(x)

