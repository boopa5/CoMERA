import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer,get_scheduler, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup,get_linear_schedule_with_warmup



# from util_optim import Adam_norm, SGD_norm
from utils import split_tensor_rank_params,init_Transformer, print_rank
from tensor_layers_CoMERA.utils import config_class
from tensor_layers_CoMERA.layers import TensorizedEmbedding, TensorizedLinear_module,  TensorizedEmbedding_order4

# from lddl.torch import get_bert_pretrain_data_loader
# from lddl.torch.utils import barrier, get_rank
# from lddl.utils import mkdir


import argparse
import logging
import numpy as np
import os
import random
import time

from datasets import load_from_disk

from tqdm import tqdm

# torch._dynamo.config.suppress_errors = True

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

def collate_fn_custom_old(batch):
    src_batch, sim_batch = [], []
    src_attn_batch = []
    seg_batch = []
    for similar,seq,seg in batch:
  
        
        src_batch.append(torch.tensor(seq))
        src_attn_batch.append(torch.tensor([1]*len(src_batch[-1])))
        # tgt_batch.append(text_transform(tgt_sample.rstrip("\n")))
        sim_batch.append(similar)
        seg_batch.append(torch.tensor(seg))

    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)

def collate_fn_custom_graph_old(batch):
    src_batch, sim_batch = [], []
    src_attn_batch = []
    seg_batch = []
    for similar,seq,seg in batch:
  
        seq_128 = torch.zeros(128)
        seq_128[:len(seq)] = seq
        
        attn = [1]*seq.shape[0] + [0]*(128-seq.shape[0])
        seg_128 = torch.tensor(seg + [0]*(128-len(seg)))
        # seg_128[:len(seg)] = torch.tensor(seg)
        
        src_batch.append(seq_128)
        src_attn_batch.append(torch.tensor(attn))
        # tgt_batch.append(text_transform(tgt_sample.rstrip("\n")))
        sim_batch.append(similar)
        seg_batch.append((seg_128))

    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)

def collate_fn_custom(batch):
    src_batch, tgt_batch,sim_batch = [], [],[]
    src_attn_batch = []
    seg_batch = []
    for c in batch:
    # for similar, src_sample, tgt_sample in batch:
        ids,attn,token_ids,similar = c['ids'],c['attn'],c['token_ids'],c['label']
        

        src_batch.append(torch.tensor(ids))
        src_attn_batch.append(torch.tensor(attn))
        # tgt_batch.append(text_transform(tgt_sample.rstrip("\n")))
        sim_batch.append(torch.tensor(similar))
        seg_batch.append(torch.tensor(token_ids))

    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)


def collate_fn_custom_graph(batch):
    src_batch, tgt_batch,sim_batch = [], [],[]
    src_attn_batch = []
    seg_batch = []
    max_len = 128
    for c in batch:
    # for similar, src_sample, tgt_sample in batch:
        ids,attn,token_ids,similar = c['ids'],c['attn'],c['token_ids'],c['label']
        
        
        diff = max_len - len(ids)
        
        
        if diff<=0:
            ids = ids[:max_len]
            attn = attn[:max_len]
            token_ids = token_ids[:max_len]

        else:
            ids = ids + [0]*diff
            attn = attn + [0]*diff
            token_ids = token_ids + [0]*diff

        src_batch.append(torch.tensor(ids))
        src_attn_batch.append(torch.tensor(attn))
        # tgt_batch.append(text_transform(tgt_sample.rstrip("\n")))
        sim_batch.append(torch.tensor(similar))
        seg_batch.append(torch.tensor(token_ids))

    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)



def ddp_setup():
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Transformer_MNLI_Trainer:
    def __init__(
        self,
        model,
        train_data,
        validation_data,
        test_data,
        optimizer,
        args
    ) -> None:
        self.args = args

        self.gpu_id = 0
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.epochs_run = args.start_epoch 
        self.snapshot_dir = args.model_dir
        
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self._build_config_forward(args)
        
        

#         if args.nv_data==0:
#             self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
#         else:
#             self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    def _build_config_forward(self,args):
        D_fwd ={
        'prune_mask':args.prune_rank,
        'threshold':1e-2,
        }
        self.config_forward = config_class(**D_fwd)
        
        

    def _eval_epoch(self,epoch):
        with torch.no_grad():
            loss_fn = self.loss_fn
            model = self.model
            model.eval()
            
            count = 0
            n_word_total = 0
            n_word_correct = 0
            n_loss = 0
            loss_total = 0
            for batch in tqdm(self.validation_data, mininterval=2,
                desc='  - (Validation)   ', leave=False):
                (target_labels,input_ids, attn_mask,seg) = (
                 batch[0].to(self.gpu_id),
                 batch[1].to(self.gpu_id),
                 batch[2].to(self.gpu_id),
                 batch[3].to(self.gpu_id)
                )


                target_labels = torch.flatten(target_labels)

                static_y_pred = model(input_ids,mask=attn_mask,seg=seg,config_forward=self.config_forward)
           

                static_loss = loss_fn(static_y_pred, target_labels)


                n_word_total += torch.sum(target_labels>=0).detach()
                n_word_correct += torch.sum(torch.argmax(static_y_pred.detach(),dim=1)==target_labels.detach())

                loss_total += static_loss.detach()*n_word_total
                n_loss += n_word_total
            
            
            if self.gpu_id == 0:
                print(" ")
                print(f'epoch {epoch}: validation result')
                
                print('loss = ', loss_total/n_loss)
                print('acc = ', n_word_correct.item()/n_word_total)
    
                

            return loss_total/n_loss,n_word_correct.item()/n_word_total
                
    
    def rank_loss(self):
        if self.args.emb_tensorized and self.args.prune_rank:
            loss = 0
            count = 0
            tol = 1e-2
            for layer in self.model.modules():
                if hasattr(layer,'tensor'):
                    for x in layer.tensor.rank_parameters:
                        count += torch.sum(x>tol)
                        loss += torch.sum(torch.nn.functional.threshold(x,tol,0))

            
            return loss/count
        else:
            return 0
    def compute_size(self):
        cur_size = 0
        for layer in self.model.modules():
            if hasattr(layer,'tensor'):
                R = [1]
                flag = True
                for x in layer.tensor.rank_parameters:
                    temp = torch.sum(x>1e-2)
                    R.append(temp)
                    flag = flag*(temp>0)
                    
                R.append(1)
                
                for i in range(len(layer.tensor.factors)):
                    cur_size += flag*layer.tensor.factors[i].shape[1]*R[i]*R[i+1]
        return cur_size
    

    def _run_epoch(self, epoch):
        
        print(f'\n Device {self.gpu_id} start epoch {epoch} without CudaGraph\n')
        count = 0
        n_word_total = 0
        n_word_correct = 0
        n_loss = 0
        loss_total = 0

        loss_fn = self.loss_fn
        optimizer = self.optimizer
        model = self.model
        model.train()

        for batch in tqdm(self.train_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            (target_labels,input_ids, attn_mask,seg) = (
             batch[0].to(self.gpu_id),
             batch[1].to(self.gpu_id),
             batch[2].to(self.gpu_id),
             batch[3].to(self.gpu_id),
            )
            
        
            optimizer.zero_grad(set_to_none=True)
            
    
            
            static_y_pred = model(input_ids,mask=attn_mask,seg=seg,config_forward=self.config_forward)

            
            
            
            
           
            static_loss = loss_fn(static_y_pred, target_labels) + 0.01*self.rank_loss()
            
            static_loss.backward()

            optimizer.step()
            
            
            n_word_total += torch.sum(target_labels>=0).detach()
            n_word_correct += torch.sum(torch.argmax(static_y_pred.detach(),dim=1)==target_labels.detach())

            loss_total += static_loss.detach()*n_word_total
            n_loss += n_word_total
            count += 1
 
            if self.gpu_id == 0 and count % self.args.print_every == 0:
                print(' ')
                
                print('count = ', count )
                print('loss = ', loss_total/n_loss)
                print('acc = ', n_word_correct.item()/n_word_total)
                
                for U in self.optimizer.param_groups:
                    print('lr = ', U['lr'])
                
                # if self.args.encoder_tensorized:
                #     print(self.model.encoder.encoder_blocks[-1].pos_ffn.fc_1.layer.tensor.rank_parameters[0])
                cursize = self.compute_size()
                print(f'current model size is {cursize}')
                print('peak memory = ', torch.cuda.max_memory_reserved()/1024/1024)
                
                loss_total = 0
                n_loss = 0
                n_word_total = 0
                n_word_correct = 0
                MLM_total = 0
                MLM_correct = 0


    def _run_epoch_late_stage(self, epoch,target_loss=0,target_size=0):
        
        print(f'\n Device {self.gpu_id} start epoch {epoch} without CudaGraph\n')
        count = 0
        n_word_total = 0
        n_word_correct = 0
        n_loss = 0
        loss_total = 0

        loss_fn = self.loss_fn
        optimizer = self.optimizer
        model = self.model
        model.train()

        for batch in tqdm(self.train_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            (target_labels,input_ids, attn_mask,seg) = (
             batch[0].to(self.gpu_id),
             batch[1].to(self.gpu_id),
             batch[2].to(self.gpu_id),
             batch[3].to(self.gpu_id),
            )
            
        
            optimizer.zero_grad(set_to_none=True)
            
    
            
            static_y_pred = model(input_ids,mask=attn_mask,seg=seg,config_forward=self.config_forward)
            cur_size = self.compute_size()
        
            model_loss = loss_fn(static_y_pred, target_labels) 
            rank_loss = self.rank_loss()
            
            static_loss = model_loss*1e-1 + rank_loss*1e-2
            
            lamb = torch.tensor(((model_loss-target_loss)/model_loss<(cur_size-target_size)/cur_size),dtype=static_loss.dtype,device=static_loss.device)
            
            static_loss = static_loss + lamb*rank_loss + (1-lamb)*model_loss
            static_loss.backward()
            optimizer.step()
            
            
            n_word_total += torch.sum(target_labels>=0).detach()
            n_word_correct += torch.sum(torch.argmax(static_y_pred.detach(),dim=1)==target_labels.detach())

            loss_total += static_loss.detach()*n_word_total
            n_loss += n_word_total
            count += 1
 
            if self.gpu_id == 0 and count % self.args.print_every == 0:
                print(' ')
                
                print('count = ', count )
                print('loss = ', loss_total/n_loss)
                print('acc = ', n_word_correct.item()/n_word_total)
                
                for U in self.optimizer.param_groups:
                    print('lr = ', U['lr'])
                
                # if self.args.encoder_tensorized:
                #     print(self.model.encoder.encoder_blocks[-1].pos_ffn.fc_1.layer.tensor.rank_parameters[0])
                cursize = self.compute_size()
                print(f'current model size is {cursize}')
                print('peak memory = ', torch.cuda.max_memory_reserved()/1024/1024)
                
                loss_total = 0
                n_loss = 0
                n_word_total = 0
                n_word_correct = 0
                MLM_total = 0
                MLM_correct = 0
    def _run_cudagraph(self,epoch,g,static_input,static_target_labels,static_attn,static_y_pred,static_loss,static_seg):

        print(f'Device {self.gpu_id} start epoch {epoch} with CudaGraph\n')

        count = 0
        n_word_total = 0
        n_word_correct = 0
        n_loss = 0
        MLM_total = 0
        MLM_correct = 0
        loss_total = 0
        

        for batch in tqdm(self.train_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            (target_labels,input_ids, attn_mask,seg) = (
             batch[0],
             batch[1],
             batch[2],
             batch[3]
            )
            # target, w1, attn ,seg = batch
            target_labels = torch.flatten(target_labels)
        
            static_input.copy_(input_ids)
            static_target_labels.copy_(target_labels)
            
            static_attn.copy_(attn_mask)
            static_seg.copy_(seg)
            

            
            g.replay()
            
            

            n_word_total += torch.sum(static_target_labels>=0).detach()
            n_word_correct += torch.sum(torch.argmax(static_y_pred.detach(),dim=1)==static_target_labels.detach())

            loss_total += static_loss.detach()*n_word_total
            n_loss += n_word_total
            count += 1
 
            if self.gpu_id == 0 and count % self.args.print_every == 0:
                print(' ')
                
                print('count = ', count )
                print('loss = ', loss_total/n_loss)
                print('acc = ', n_word_correct.item()/n_word_total)
                
                for U in self.optimizer.param_groups:
                    print('lr = ', U['lr'])
                
                # if self.args.encoder_tensorized:
                #     print(self.model.encoder.encoder_blocks[-1].pos_ffn.fc_1.layer.tensor.rank_parameters[0])
                cursize = self.compute_size()
                print(f'current model size is {cursize}')
                print('peak memory = ', torch.cuda.max_memory_reserved()/1024/1024)
                
                loss_total = 0
                n_loss = 0
                n_word_total = 0
                n_word_correct = 0
                MLM_total = 0
                MLM_correct = 0
                
               




    def _build_graph(self,model=None):
        args = self.args
        device = f'cuda:{self.gpu_id}'
        optimizer = self.optimizer
        if model==None:
            model = self.model
        loss_fn = self.loss_fn

        for U in optimizer.param_groups:
            U['capturable'] = True

        static_input = torch.randint(30000,(args.batch_size, args.max_len), device=device)
        static_target_labels = torch.randint(0,3,(args.batch_size,), device=device)
        
        static_attn = torch.ones((args.batch_size,args.max_len), dtype=torch.int, device=device)
        
        static_seg = torch.ones((args.batch_size,args.max_len), dtype=torch.int, device=device)

        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        
        model = self.model
        
        with torch.cuda.device(device):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.stream(s):
                
                for i in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    # with torch.autocast(device_type="cuda"):
                    static_y_pred = model(static_input,mask=static_attn,seg=static_seg,config_forward=self.config_forward)
    #                 static_y_pred = torch.flatten(static_y_pred_temp,start_dim=0, end_dim=1)

                    static_loss = loss_fn(static_y_pred, static_target_labels) + 0.01*self.rank_loss()
                    static_loss.backward()

                    optimizer.step()

            torch.cuda.current_stream(device).wait_stream(s)
            g = torch.cuda.CUDAGraph()

            c = 20

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.graph(g,stream=s):
                # with torch.autocast(device_type="cuda"):
                static_y_pred = model(static_input,mask=static_attn,seg=static_seg,config_forward=self.config_forward)
    #             static_y_pred = torch.flatten(static_y_pred,start_dim=0, end_dim=1)

                static_loss = loss_fn(static_y_pred, static_target_labels) + 0.01*self.rank_loss()
                static_loss.backward()
                optimizer.step()

        return g,static_input,static_target_labels,static_attn,static_y_pred,static_loss,static_seg

    def _build_graph_late_stage(self,model=None,target_size=None,target_loss=None):
        args = self.args
        device = f'cuda:{self.gpu_id}'
        optimizer = self.optimizer
        if model==None:
            model = self.model
        loss_fn = self.loss_fn

        for U in optimizer.param_groups:
            U['capturable'] = True

        static_input = torch.randint(30000,(args.batch_size, args.max_len), device=device)
        static_target_labels = torch.randint(0,3,(args.batch_size,), device=device)
        
        static_attn = torch.ones((args.batch_size,args.max_len), dtype=torch.int, device=device)
        
        static_seg = torch.ones((args.batch_size,args.max_len), dtype=torch.int, device=device)

        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        
        model = self.model
        
        with torch.cuda.device(device):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.stream(s):
                for i in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    # with torch.autocast(device_type="cuda"):
                    static_y_pred = model(static_input,mask=static_attn,seg=static_seg,config_forward=self.config_forward)
    #                 static_y_pred = torch.flatten(static_y_pred_temp,start_dim=0, end_dim=1)
                    cur_size = self.compute_size()
            
                    model_loss = loss_fn(static_y_pred, static_target_labels) 
                    rank_loss = self.rank_loss()
                    
                    static_loss = model_loss*1e-1 + rank_loss*1e-2
                    
                    lamb = torch.tensor(((model_loss-target_loss)/model_loss<(cur_size-target_size)/cur_size),dtype=static_loss.dtype,device=static_loss.device)
                    
                    static_loss = static_loss + lamb*rank_loss + (1-lamb)*model_loss
                
                    static_loss.backward()

                    optimizer.step()

            torch.cuda.current_stream(device).wait_stream(s)
            g = torch.cuda.CUDAGraph()

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.graph(g,stream=s):
                static_y_pred = model(static_input,mask=static_attn,seg=static_seg,config_forward=self.config_forward)
                cur_size = self.compute_size()
            
                model_loss = loss_fn(static_y_pred, static_target_labels) 
                rank_loss = self.rank_loss()
                
                static_loss = model_loss*1e-1 + rank_loss*1e-2
                
                lamb = torch.tensor(((model_loss-target_loss)/model_loss<(cur_size-target_size)/cur_size),dtype=static_loss.dtype,device=static_loss.device)
                
                static_loss = static_loss + lamb*rank_loss + (1-lamb)*model_loss
                static_loss.backward()
                optimizer.step()

        return g,static_input,static_target_labels,static_attn,static_y_pred,static_loss,static_seg


        

    def train(self):
        
        print(self.model)
            
        best_acc = 0
        path = None
        
        max_epochs = self.args.epochs_earlystage+self.args.epochs_latestage
        
        use_cudagraph = self.args.use_cuda_graph
        if use_cudagraph==True:
            self.model = torch.compile(self.model)
            if self.args.model_loadPATH != None:
                self.model.load_state_dict(torch.load(self.args.model_loadPATH))
            torch.save(self.model.state_dict(),'temp_cudagraph.chkpt')
            torch.save(self.optimizer.state_dict(),'temp_optim_cudagraph.chkpt')
            
            static_tensors = self._build_graph()
            self.model.load_state_dict(torch.load('temp_cudagraph.chkpt'))
            self.optimizer.load_state_dict(torch.load('temp_optim_cudagraph.chkpt'))
            os.remove('temp_cudagraph.chkpt')
            os.remove('temp_optim_cudagraph.chkpt')

        for epoch in range(self.epochs_run, max_epochs):
            if use_cudagraph and epoch == self.args.epochs_earlystage:
                print('start late stage cudagprah processing')
                
                static_tensors[0].reset()
                target_size = self.args.target_ratio*self.compute_size()
                torch.save(self.model.state_dict(),'temp_cudagraph.chkpt')
                torch.save(self.optimizer.state_dict(),'temp_optim_cudagraph.chkpt')
                static_tensors = self._build_graph_late_stage(target_loss=0.8,target_size=target_size)
                self.model.load_state_dict(torch.load('temp_cudagraph.chkpt'))
                self.optimizer.load_state_dict(torch.load('temp_optim_cudagraph.chkpt'))
                os.remove('temp_cudagraph.chkpt')
                os.remove('temp_optim_cudagraph.chkpt')

            print('start epoch = ',epoch)
            t1 = time.time()
            if use_cudagraph==False:
                self._run_epoch(epoch)
                t2 = time.time()
            else:
                self._run_cudagraph(epoch,*static_tensors)
                t2 = time.time()

            loss,acc = self._eval_epoch(epoch)
            if self.gpu_id == 0:
                for U in self.optimizer.param_groups:
                    print('lr = ', U['lr'])
                if self.args.encoder_tensorized and self.args.prune_rank:
                    print_rank(self.model,1e-2,6)
                    cursize = self.compute_size()
                    print(f'current model size is {cursize}')
                if acc>best_acc:
                    best_acc = acc
                    if path!=None:
                        os.remove(path)
                    path = self.snapshot_dir + f'MNLI_TensorEn{self.args.encoder_tensorized}_Emb{self.args.emb_tensorized}_lr{self.args.lr_origin}_{self.args.lr_tensor}_acc{acc:.4f}.chkpt'
                    torch.save(self.model.state_dict(),path)
                    
                # self._save_snapshot(path)
           
            print(f'time for training epoch {epoch} is {t2-t1}')



def prepare_dataloader(args):

    data_path='./datasets/MNLI_len128'
    
    dataset = load_from_disk(data_path)
    if args.use_cuda_graph:
        collate_train = collate_fn_custom_graph
    else:
        collate_train = collate_fn_custom
    collate_test = collate_fn_custom
    
    
    training_data = torch.utils.data.DataLoader(dataset['train'],batch_size = args.batch_size,shuffle=True,drop_last=True,collate_fn=collate_train)
    validation_data = torch.utils.data.DataLoader(dataset['validation_matched'],batch_size = 32,shuffle=False,drop_last=True,collate_fn=collate_test)
    test_data = torch.utils.data.DataLoader(dataset['test_matched'],batch_size = 32,shuffle=False,drop_last=True,collate_fn=collate_test)

    return training_data,validation_data,test_data





def main(args):
    model = init_Transformer(emb_tensorized=args.emb_tensorized,encoder_tensorized=args.encoder_tensorized,r=args.max_rank,num_class=3,n_layers=args.n_layers)
    # if args.model_loadPATH != None:
    #     model.load_state_dict(torch.load(args.model_loadPATH))
    
    
    train_data,validation_data,test_data = prepare_dataloader(args)

    par_tensor, par_origin,par_rank = split_tensor_rank_params(model)
    print(len(par_origin))
    print(len(par_tensor))
    print(len(par_rank))
    
    
    if args.prune_rank:
        optimizer = torch.optim.AdamW([{'params':par_tensor,'lr':args.lr_tensor,'weight_decay':0},
                            {'params':par_origin,'lr':args.lr_origin,'weight_decay':0},
                                {'params':par_rank,'lr':args.lr_rank,'weight_decay':0}])
    else:
        optimizer = torch.optim.AdamW([{'params':par_tensor,'lr':args.lr_tensor,'weight_decay':0},
                            {'params':par_origin,'lr':args.lr_origin,'weight_decay':0}])
           
       
        

    

    
    trainer = Transformer_MNLI_Trainer(model, train_data, validation_data,test_data,optimizer, args)
    

    trainer.train()



if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=128)

    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--epochs-earlystage', type=int, default=20)
    parser.add_argument('--epochs-latestage', type=int, default=10)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--target-ratio', type=float, default=0.5)
    parser.add_argument('--print-every', type=int, default=500)
    

  
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--log-freq', type=int, default=1000)
    parser.add_argument('--log-dir', type=str, default=None)


  
    parser.add_argument('--model-dir',type=str)
  
  
    parser.add_argument('--lr-tensor', type=float, default=1e-4)
    parser.add_argument('--lr-origin', type=float, default=1e-4)
    parser.add_argument('--lr-rank', type=float, default=1e-3)
    
    parser.add_argument('--model-loadPATH', type=str, default=None)
    parser.add_argument('--n-layers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--use-cuda-graph', type=int, default=0)

    parser.add_argument('--max-rank', type=int, default=100)
    parser.add_argument('--emb-tensorized', type=int, default=0)
    parser.add_argument('--encoder-tensorized', type=int, default=1)


    parser.add_argument('--prune-rank', type=int, default=0)

    
    
    
    
    
    




    args = parser.parse_args()
    
    # if args.scheduler=='constant':
    #     args.scheduler = args.scheduler + '_with_warmup'

    args.use_cuda_graph = (args.use_cuda_graph==1)
    args.emb_tensorized = (args.emb_tensorized==1)
    args.encoder_tensorized = (args.encoder_tensorized==1)
    args.prune_rank = (args.prune_rank==1)
    
    print(args)

    
    main(args)