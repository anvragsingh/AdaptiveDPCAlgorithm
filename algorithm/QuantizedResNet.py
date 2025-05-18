import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, 
                 initial_bits=8, min_bits=4):
        super(QuantizedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight with full precision
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.initial_bits = initial_bits
        self.min_bits = min_bits
        self.current_bits = initial_bits
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input):
        # Quantize weights before forward pass
        if self.current_bits < 32:
            scale = (self.weight.max() - self.weight.min()) / (2**self.current_bits - 1)
            zero_point = (-self.weight.min() / scale).round().clamp(0, 2**self.current_bits - 1)
            q_weight = ((self.weight / scale) + zero_point).round().clamp(0, 2**self.current_bits - 1)
            dq_weight = (q_weight - zero_point) * scale
        else:
            dq_weight = self.weight
            
        return F.conv2d(input, dq_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def set_bit_width(self, bits):
        self.current_bits = max(self.min_bits, min(self.initial_bits, bits))

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 initial_bits=8, min_bits=4):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight with full precision
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.initial_bits = initial_bits
        self.min_bits = min_bits
        self.current_bits = initial_bits
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input):
        # Quantize weights before forward pass
        if self.current_bits < 32:
            scale = (self.weight.max() - self.weight.min()) / (2**self.current_bits - 1)
            zero_point = (-self.weight.min() / scale).round().clamp(0, 2**self.current_bits - 1)
            q_weight = ((self.weight / scale) + zero_point).round().clamp(0, 2**self.current_bits - 1)
            dq_weight = (q_weight - zero_point) * scale
        else:
            dq_weight = self.weight
            
        return F.linear(input, dq_weight, self.bias)
    
    def set_bit_width(self, bits):
        self.current_bits = max(self.min_bits, min(self.initial_bits, bits))

def make_quantized_resnet(arch, block, layers, num_classes=1000, initial_bits=8, min_bits=4):
    """Create a quantized ResNet model"""
    model = ResNet(block, layers, num_classes)
    
    # Replace Conv2d and Linear layers with quantized versions
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_conv = QuantizedConv2d(
                module.in_channels, module.out_channels, 
                module.kernel_size, module.stride,
                module.padding, module.dilation, 
                module.groups, module.bias is not None,
                initial_bits, min_bits
            )
            new_conv.weight.data = module.weight.data
            if module.bias is not None:
                new_conv.bias.data = module.bias.data
            setattr(model, name, new_conv)
        elif isinstance(module, nn.Linear):
            new_linear = QuantizedLinear(
                module.in_features, module.out_features,
                module.bias is not None,
                initial_bits, min_bits
            )
            new_linear.weight.data = module.weight.data
            if module.bias is not None:
                new_linear.bias.data = module.bias.data
            setattr(model, name, new_linear)
    
    return model