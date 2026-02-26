"""
Differentiable Logic Layer implementation.
Pure PyTorch implementation (no CUDA dependency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .functional import bin_op_s, GradFactor


def create_random_onehot_tensor(n, m=16):
    """Create random one-hot tensor for fixed gates initialization."""
    indices = torch.randint(0, m, (n,))
    return F.one_hot(indices, num_classes=m).float()


class LogicLayer(nn.Module):
    """
    Differentiable logic gate layer.
    
    Each neuron computes a soft combination of 16 possible binary logic operations
    on two input concepts.
    """
    
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            connections: str = 'random',
            fixed_gates: bool = False,
            n_logic_gates: int = 16,
            concept_pairs: torch.Tensor = None
    ):
        """
        Args:
            in_dim: Input dimensionality (number of concepts)
            out_dim: Output dimensionality (number of logic neurons)
            device: Device to use ('cuda' or 'cpu')
            grad_factor: Gradient scaling factor (for deep networks)
            connections: How to connect inputs ('random', 'correlated')
            fixed_gates: If True, use fixed random gates instead of learned
            n_logic_gates: Number of logic gate types (default 16)
            concept_pairs: Pre-defined concept pairs for 'correlated' connections
                          Shape: (out_dim, 2) where each row is (idx_a, idx_b)
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.n_logic_gates = n_logic_gates
        self.fixed_gates = fixed_gates
        self.connections = connections
        self.concept_pairs = concept_pairs
        
        # Initialize gate weights
        if self.fixed_gates:
            # Fixed random gates (not learned)
            self.weights = nn.Parameter(
                create_random_onehot_tensor(self.out_dim, self.n_logic_gates).to(device),
                requires_grad=False
            )
        else:
            # Learnable gate weights
            self.weights = nn.Parameter(
                torch.randn(out_dim, self.n_logic_gates, device=device)
            )
        
        # Initialize connections
        self.indices = self._get_connections(connections, device)
    
    def _get_connections(self, connections, device):
        """Get input connection indices for each neuron."""
        
        if connections == 'random':
            # Random connections
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            
        elif connections == 'correlated':
            # Use pre-defined concept pairs
            if self.concept_pairs is None:
                raise ValueError("concept_pairs must be provided for 'correlated' connections")
            a = self.concept_pairs[:, 0]
            b = self.concept_pairs[:, 1]
            
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        
        a = a.to(torch.int64).to(device)
        b = b.to(torch.int64).to(device)
        
        return a, b
    
    def forward(self, x):
        """
        Forward pass through the logic layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
               Values should be in [0, 1] (soft binary)
        
        Returns:
            Output tensor of shape (batch_size, out_dim)
        """
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)
        
        # Get inputs for each neuron
        a_idx, b_idx = self.indices
        a = x[..., a_idx]  # (batch_size, out_dim)
        b = x[..., b_idx]  # (batch_size, out_dim)
        
        if self.training:
            # Soft gate selection during training
            if self.fixed_gates:
                gate_weights = self.weights
            else:
                gate_weights = F.softmax(self.weights, dim=-1)
            out = bin_op_s(a, b, gate_weights)
        else:
            # Hard gate selection during inference
            hard_weights = F.one_hot(
                self.weights.argmax(-1), 
                self.n_logic_gates
            ).float()
            out = bin_op_s(a, b, hard_weights)
        
        return out.to(torch.float32)
    
    def extra_repr(self):
        return f'in={self.in_dim}, out={self.out_dim}, connections={self.connections}'


class GroupSum(nn.Module):
    """
    Group sum module for aggregating logic neuron outputs.
    Divides outputs into k groups and sums within each group.
    """
    
    def __init__(self, k: int, tau: float = 1.):
        """
        Args:
            k: Number of output groups (e.g., number of classes)
            tau: Temperature for scaling the output
        """
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, f"Input dim {x.shape[-1]} not divisible by k={self.k}"
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return f'k={self.k}, tau={self.tau}'
