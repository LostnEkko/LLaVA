import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import random
import math
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class PadByRepeat(nn.Module):
    def __init__(self, mm_aligner_size, output_size):
        super().__init__()
        self.mm_aligner_size = mm_aligner_size
        self.output_size = output_size

    def forward(self, x):
        #TODO: This line ignores the fact output_size can be different than aligner_size as temporary solution
        return x.repeat(1, 1, self.output_size // self.mm_aligner_size)

# 16, 24, 30, works only for 3 dim tensor
class BranchProjectionWithRepeatedPad(nn.Module):
    def __init__(self, pad_token_size_base: int, in_features: int, out_features: int):
        super().__init__()
        self.pad_token_size_base = pad_token_size_base
        self.in_features = in_features
        self.out_features = out_features

        self.base_projection_sizes = [16, 24, 30]
        self.base_projection_num = len(self.base_projection_sizes)

        self.max_base_size = self.out_features // self.pad_token_size_base

        # 0 indexed for pading weights and bias
        self.weight_0 = Parameter(torch.empty((pad_token_size_base, in_features)))
        self.bias_0 = Parameter(torch.empty(pad_token_size_base))

        for idx, base_size in enumerate(self.base_projection_sizes):
            setattr(self, "weight_" + str(idx + 1), Parameter(torch.empty((base_size * pad_token_size_base, in_features))))
            setattr(self, "bias_" + str(idx + 1), Parameter(torch.empty(base_size * pad_token_size_base)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # init like pytorch linear layer, adapted from nn.Linear
        for idx in range(self.base_projection_num + 1):
            cur_weight = "weight_" + str(idx)
            cur_bias = "bias_" + str(idx)
            init.kaiming_uniform_(getattr(self, cur_weight), a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self, cur_weight))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(getattr(self, cur_bias), -bound, bound)

    def forward(self, x):
        idx = random.randint(0, self.base_projection_num - 1) if self.weight_0.requires_grad else (self.base_projection_num - 1)
        print("idx", idx)
        # idx = self.base_projection_num - 1
        weight = getattr(self, "weight_" + str(idx + 1))
        bias = getattr(self, "bias_" + str(idx + 1))
        pad_multiplier = self.max_base_size - self.base_projection_sizes[idx]
        weight = torch.cat((weight, self.weight_0.repeat(pad_multiplier, 1)), dim=0)
        bias = torch.cat((bias, self.bias_0.repeat(pad_multiplier)), dim=0)
        return nn.functional.linear(x, weight, bias)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projector_aligner(config, mm_aligner_size, output_size=None):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)

    modules = []
    if projector_type == 'linear':
        return modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
    
    elif mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
    
    output_size = config.hidden_size if output_size is None else output_size 
    modules.append(nn.GELU())
    # 32*128 version
    # modules.append(nn.Linear(config.hidden_size, mm_aligner_size))
    # modules.append(PadByRepeat(mm_aligner_size, config.hidden_size))
    # prjection with different size plus padding version
    modules.append(BranchProjectionWithRepeatedPad(mm_aligner_size, config.hidden_size, output_size))
    return nn.Sequential(*modules)


    
