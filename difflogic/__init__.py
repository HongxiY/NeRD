"""
Differentiable Logic Gates - Local Implementation
No CUDA dependency, pure PyTorch.
"""

from .difflogic import LogicLayer, GroupSum
from .functional import bin_op, bin_op_s

__all__ = ['LogicLayer', 'GroupSum', 'bin_op', 'bin_op_s']
