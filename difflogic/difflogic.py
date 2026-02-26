import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .functional import bin_op_s, GradFactor

EXCLUDED_GATES = [6, 9]


def create_random_onehot_tensor(n, m=16, excluded_gates=None):
    if excluded_gates is None:
        excluded_gates = []

    valid_gates = [i for i in range(m) if i not in excluded_gates]

    indices = torch.tensor([valid_gates[i] for i in torch.randint(0, len(valid_gates), (n,))])
    return F.one_hot(indices, num_classes=m).float()


class LogicLayer(nn.Module):
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
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.n_logic_gates = n_logic_gates
        self.fixed_gates = fixed_gates
        self.connections = connections
        self.concept_pairs = concept_pairs
        self.excluded_gates = EXCLUDED_GATES

        if self.fixed_gates:
            self.weights = nn.Parameter(
                create_random_onehot_tensor(self.out_dim, self.n_logic_gates, self.excluded_gates).to(device),
                requires_grad=False
            )
        else:
            init_weights = torch.randn(out_dim, self.n_logic_gates, device=device)

            for gate_idx in self.excluded_gates:
                init_weights[:, gate_idx] = -1e9  # Use large negative instead of -inf for numerical stability
            self.weights = nn.Parameter(init_weights)

            self.weights.register_hook(self._zero_excluded_gradients)

        self.indices = self._get_connections(connections, device)

    def _zero_excluded_gradients(self, grad):
        grad = grad.clone()
        for gate_idx in self.excluded_gates:
            grad[:, gate_idx] = 0
        return grad

    def _get_connections(self, connections, device):
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]

        elif connections == 'correlated':
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
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)

        # Get inputs for each neuron
        a_idx, b_idx = self.indices
        a = x[..., a_idx]  # (batch_size, out_dim)
        b = x[..., b_idx]  # (batch_size, out_dim)

        if self.training:
            if self.fixed_gates:
                gate_weights = self.weights
            else:
                gate_weights = F.softmax(self.weights, dim=-1)
            out = bin_op_s(a, b, gate_weights)
        else:
            hard_weights = F.one_hot(
                self.weights.argmax(-1),
                self.n_logic_gates
            ).float()
            out = bin_op_s(a, b, hard_weights)

        return out.to(torch.float32)

    def get_gate_distribution(self):
        with torch.no_grad():
            gate_probs = F.softmax(self.weights, dim=-1)
            gate_types = self.weights.argmax(-1)
            return gate_types, gate_probs

    def extra_repr(self):
        return f'in={self.in_dim}, out={self.out_dim}, connections={self.connections}, excluded_gates={self.excluded_gates}'


class GroupSum(nn.Module):
    def __init__(self, k: int, tau: float = 1.):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, f"Input dim {x.shape[-1]} not divisible by k={self.k}"
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return f'k={self.k}, tau={self.tau}'
