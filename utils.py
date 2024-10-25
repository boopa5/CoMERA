from transformers import AutoTokenizer, BertModel
import torch


from tensor_layers_CoMERA.layers import TensorizedLinear_module
import math


from tensor_layers_CoMERA.utils import config_class
from tensor_layers_CoMERA.Transformer_tensor import Transformer_classification
# from tensor_layers_CoMERA.GPT_tensor import GPT_pretrain, GPT_LM




def split_galore_params(model):
    p_galore = []
    p_other = []
    for name,p in model.named_parameters():
        if 'linear' in name:
            p_galore.append(p)
        else:
            p_other.append(p)
    return p_galore,p_other
        


def split_tensor_params(model):
    par_all = model.parameters()
    for p in par_all:
        p.requires_grad = True
    for layer in model.modules():
        if hasattr(layer,'tensor'):
            if len(layer.tensor.factors)==6:
                for U in layer.tensor.factors:
                    U.requires_grad = False
    par_tensor = []
    par_origin = []

    for p in model.parameters():
        if p.requires_grad == True:
            par_origin.append(p)
        else:
            p.requires_grad = True
            par_tensor.append(p)
    
    return par_tensor, par_origin


def split_tensor_rank_params(model):

    par_tensor = []
    par_origin = []
    par_rank = []

    for name,par in model.named_parameters():
        if 'encoder' in name and 'tensor.factors' in name:
            par_tensor.append(par)
        elif 'encoder' in name and 'tensor.rank_parameters' in name:
            par_rank.append(par)
        else:
            par_origin.append(par)
    
    return par_tensor, par_origin, par_rank
    



def rank_penalty(model,tol):
    loss = 0
    count = 0
    for layer in model.modules():
        if hasattr(layer,'tensor'):
            for x in layer.tensor.rank_parameters:
                loss += torch.sum(torch.abs(x[x>tol]))
                count += torch.sum(x>tol)
    return loss/count


    
def set_rank_par(model):
    loss = 0
    count = 0
    for layer in model.modules():
        if hasattr(layer,'tensor'):
            for x in layer.tensor.rank_parameters:
                x.data[:] = 1.0

    

def print_rank(transformer,tol,n_layers=12):
    for i in range(n_layers):
        block = transformer.encoder.encoder_blocks[i]
        attn = block.slf_attn
        pff = block.pos_ffn
        
        print("")
        print('layer = ',i)
        print('Q rank = ',attn.w_qs.layer.tensor.estimate_rank(tol))
        print('K rank = ',attn.w_ks.layer.tensor.estimate_rank(tol))
        print('V rank = ',attn.w_vs.layer.tensor.estimate_rank(tol))
        print('FC rank = ',attn.fc.layer.tensor.estimate_rank(tol))
        print('PFF 1 rank = ',pff.fc_1.layer.tensor.estimate_rank(tol))
        print('PFF 2 rank = ',pff.fc_2.layer.tensor.estimate_rank(tol))
        
def get_rank_parameters(model):
    par_rank = []
    for layer in model.modules():
        if hasattr(layer,'tensor'):
            par_rank += list(layer.tensor.rank_parameters)
            
    par_model = []
    for p in par_rank:
        p.requires_grad = False
    for p in model.parameters():
        if p.requires_grad == True:
            par_model.append(p)
            
    for p in model.parameters():
        p.requires_grad = True
        
    
    return par_rank, par_model








def init_Transformer(emb_tensorized=False,encoder_tensorized=True,r=100,num_class=3,n_layers=6):
   
    D = {
        'n_layers': n_layers,
        'vocab_size': 30522,
        'n_position': 512,
        'd_model':768,
        'd_hid':768*4,
        'n_head':12,
        'tensorized':encoder_tensorized,
        'emb_tensorized':emb_tensorized,
        'dropout': 0.1,
        'embedding': None,
        'classification': None,
        'pff': {},
        'attn': {}
        }
    build_rank_parameters = True
    set_scale_factors = False

    emb_shape = [[16,20,10,10],[4,4,8,6]]
    emb_rank = r
    emb_rank = [1,32,32,32,1]


    attn_shape = [12,8,8,8,8,12]
    attn_rank = [1,12,r,r,r,12,1]

    pff_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
    pff_rank = [[1,12,r,r,r,16,1],[1,16,r,r,r,12,1]]

    # attn_shape = [768,768]
    # attn_rank = [1,r,1]

    # pff_shape = [[768,768*4],[768*4,768]]
    # pff_rank = [[1,r,1],[1,r,1]]

    classification_shape = [12,8,8,8,8,12]
    classification_rank = [1,12,r,r,r,12,1]


    config_model =config_class(**D)

    config_model.pff[0] = config_class(shape=pff_shape[0],ranks=pff_rank[0],set_scale_factors=set_scale_factors,build_rank_parameters=build_rank_parameters)
    config_model.pff[1] = config_class(shape=pff_shape[1],ranks=pff_rank[1],set_scale_factors=set_scale_factors,build_rank_parameters=build_rank_parameters)


    config_attn_sublayers = config_class(shape=attn_shape,ranks=attn_rank,set_scale_factors=set_scale_factors,build_rank_parameters=build_rank_parameters)
    for key in ['q','k','v','fc']:
        config_model.attn[key] = config_attn_sublayers


    config_model.embedding = config_class(shape=emb_shape,ranks=emb_rank,set_scale_factors=set_scale_factors)


    config_classification = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=num_class,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors,build_rank_parameters=build_rank_parameters)


    transformer = Transformer_classification(config_model,config_classification)

    return transformer
