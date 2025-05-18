import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class AdaptiveDPC:
    def __init__(self, model, initial_bits=8, min_bits=4, delta=0.5):
        """
        Initialize Adaptive DPC quantizer
        
        Args:
            model: PyTorch model to quantize
            initial_bits: Starting bit-width (B_max)
            min_bits: Minimum bit-width (B_min)
            delta: Precision interval parameter
        """
        self.model = model
        self.initial_bits = initial_bits
        self.min_bits = min_bits
        self.delta = delta
        self.layer_capacities = self._calculate_layer_capacities()
        self.bit_widths = self._initialize_bit_widths()
        
    def _calculate_layer_capacities(self):
        """Calculate capacity for each layer (number of parameters)"""
        capacities = OrderedDict()
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_name = name.replace('.weight', '')
                capacities[layer_name] = param.numel()
        return capacities
    
    def _initialize_bit_widths(self):
        """Initialize bit-widths based on layer capacities"""
        max_cap = max(self.layer_capacities.values())
        min_cap = min(self.layer_capacities.values())
        
        bit_widths = OrderedDict()
        for name, capacity in self.layer_capacities.items():
            # Logarithmic schedule from paper
            normalized_cap = (capacity - min_cap) / (max_cap - min_cap + 1e-8)
            bit_width = self.initial_bits - self.delta * np.log2(1 + normalized_cap * (2**((self.initial_bits - self.min_bits)/self.delta) - 1))
            bit_width = max(self.min_bits, min(self.initial_bits, bit_width))
            bit_widths[name] = int(round(bit_width))
        
        return bit_widths
    
    def quantize_weights(self):
        """Quantize weights based on current bit-widths"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_name = name.replace('.weight', '')
                bits = self.bit_widths[layer_name]
                q_param = self.quantize_tensor(param.data, bits)
                param.data = q_param
    
    def quantize_tensor(self, tensor, bits):
        """Quantize tensor to specified bit-width"""
        if bits >= 32:  # No quantization
            return tensor
            
        # Calculate scale and zero-point
        v_max = tensor.max()
        v_min = tensor.min()
        scale = (v_max - v_min) / (2**bits - 1)
        zero_point = (-v_min / scale).round().clamp(0, 2**bits - 1)
        
        # Quantize and dequantize
        q_tensor = ((tensor / scale) + zero_point).round().clamp(0, 2**bits - 1)
        dq_tensor = (q_tensor - zero_point) * scale
        
        return dq_tensor
    
    def update_precision(self, gradient_stats):
        """
        Adaptively update precision based on gradient statistics
        
        Args:
            gradient_stats: Dictionary containing gradient information for each layer
        """
        for name in self.bit_widths.keys():
            # Example adaptation rule - can be customized
            grad_norm = gradient_stats[name]['norm']
            grad_var = gradient_stats[name]['var']
            
            # If gradients are small and stable, try reducing precision
            if grad_norm < 1e-4 and grad_var < 1e-8:
                self.bit_widths[name] = max(self.min_bits, self.bit_widths[name] - 1)
            # If gradients are large or noisy, consider increasing precision
            elif grad_norm > 1e-2 or grad_var > 1e-4:
                self.bit_widths[name] = min(self.initial_bits, self.bit_widths[name] + 1)