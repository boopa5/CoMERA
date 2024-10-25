import torch
import torch.nn as nn
import numpy as np

from .Transformer_tensor_sublayers import EncoderLayer, Transformer_Embedding, Transformer_classifier
from .layers import TensorizedEmbedding, TensorizedLinear_module, wrapped_linear_layers


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print(self.pos_table[:, :x.size(1),:2])
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embedding = Transformer_Embedding(config)

        self.encoder_blocks = nn.ModuleList()

        for i in range(config.n_layers):
            self.encoder_blocks.append(EncoderLayer(config))
    
    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.embedding(input,seg=seg,config_forward=config_forward)

        for layer in self.encoder_blocks:
            output, attn = layer(output,mask=mask,config_forward=config_forward)
        
        return output

class Transformer_classification(nn.Module):
    def __init__(self, config, config_classifiction):
        super(Transformer_classification, self).__init__()

        self.encoder = Encoder(config)

        self.classifier = Transformer_classifier(config_classifiction)

    def forward(self,input,mask=None,seg=None,config_forward=None):
        output = self.encoder(input,mask=mask,seg=seg,config_forward=config_forward)

        output = output[:,0,:]

        output = self.classifier(output,config_forward=config_forward)

        return output

