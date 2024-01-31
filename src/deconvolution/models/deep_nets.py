import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from deconvolution.models.utils import get_activation_function, transformer_timestep_embedding

#...Multi-Layer Perceptron architecture:

class MLP(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.device = configs.DEVICE
        self.define_deep_models(configs)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, configs):
        self.dim_input = configs.DIM_INPUT
        self.dim_output = configs.DIM_INPUT
        self.dim_hidden = configs.DIM_HIDDEN 
        self.dim_time_emb = configs.DIM_TIME_EMB if configs.DIM_TIME_EMB is not None else 1
        self.num_layers = configs.NUM_LAYERS
        self.act_fn = get_activation_function(configs.ACTIVATION)

        layers = [nn.Linear(self.dim_input + self.dim_time_emb, self.dim_hidden)]

        for _ in range(self.num_layers - 1): 
            layers.append(nn.Linear(self.dim_hidden, self.dim_hidden))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(self.dim_hidden, self.dim_output)

    def forward(self, t, x, context=None, mask=None):

        x = x.to(self.device)
        time_embeddings = transformer_timestep_embedding(t.squeeze(1), embedding_dim=self.dim_time_emb) if t is not None else t
        time_embeddings = time_embeddings.to(self.device)
        x = torch.concat([x, time_embeddings], dim=1)
        
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output_layer(x)
        return x

    def init_weights(self):
        for layer in self.layers + [self.output_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
#...ResNet architecture:

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, num_layers_per_block=2):
        super(ResidualBlock, self).__init__()

        layers = []
        for _ in range(num_layers_per_block):
            layers.extend([torch.nn.Linear(dim, dim), torch.nn.SELU()])
        self.block = torch.nn.Sequential(*layers[:-1])

    def forward(self, x):
        return x + self.block(x)
    
class _ResNet(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, num_blocks=3, num_layers_per_block=2):
        super(_ResNet, self).__init__()

        if out_dim is None:
            out_dim = dim

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU()
        )
        self.blocks = torch.nn.ModuleList([ResidualBlock(w, num_layers_per_block) for _ in range(num_blocks)])
        self.output_layer = torch.nn.Linear(w, out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class ResNet(nn.Module):
    ''' Wrapper class for the ResNet architecture
    '''
    def __init__(self, configs):
        super(ResNet, self).__init__()
        self.device = configs.DEVICE
        self.resnet = _ResNet(dim=configs.DIM_INPUT, 
                              out_dim=None,
                              w=configs.DIM_HIDDEN, 
                              num_blocks=configs.NUM_BLOCKS,
                              num_layers_per_block=configs.NUM_BLOCK_LAYERS,
                              time_varying=True)
                
    def forward(self, t, x, context=None, mask=None, sampling=False):
        x = torch.cat([x, t], dim=-1)
        x = x.to(self.device)
        self.resnet = self.resnet.to(self.device)
        return self.resnet.forward(x)
