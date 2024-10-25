import torch
import numpy as np
# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
import torch.nn.functional as F

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))






    
    

class TT_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, *factors):

        with torch.no_grad():

            tt_shape = [U.shape[1] for U in factors]
            ndims = len(factors)
            d = int(ndims / 2)

            ctx.d = d
            ctx.input_shape = matrix.shape
            # if len(matrix.shape)==3:
            #     out_shape = [matrix.shape[0],matrix.shape[1],np.prod(list(tt_shape[d:]))]
            #     matrix = torch.flatten(matrix,start_dim=0,end_dim=1)
            # else:
            #     out_shape = [matrix.shape[0],np.prod(list(tt_shape[d:]))]
            B = len(matrix.shape) - 1
       
            out_shape = list(matrix.shape[:B]) + [np.prod(list(tt_shape[d:]))]
            ctx.out_shape = out_shape
            
            # print(matrix.shape)
            matrix = torch.flatten(matrix,start_dim=0,end_dim=B-1)
            # print(B)
            # print(matrix.shape)
            
            saved_tensors = [matrix] + list(factors)
            

            # ctx.factors = factors
            # ctx.matrix = matrix


            
    
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            matrix_cols = matrix.shape[0]

            left = []
            right = []

            
            output = factors[0].reshape(-1, ranks[1])
            left.append(output)
            

            for core in factors[1:d]:
                output = (torch.tensordot(output, core, dims=([-1], [0])))
                left.append(output)

            output = F.linear(matrix, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0))

            saved_tensors.append(output)

            saved_tensors = saved_tensors + left

            temp = factors[d]
            right.append(temp)
            for core in factors[d + 1:]:
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                right.append(temp)

            right[-1] = torch.squeeze(right[-1])

            
            output = F.linear(output, torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
                                            0, -1)).reshape(matrix_cols, np.prod(tt_shape_col)).reshape(*out_shape)

        
            
            saved_tensors = saved_tensors + right

            ctx.save_for_backward(*saved_tensors)
       
   
        return output

       
    @staticmethod
    def backward(ctx, dy):
        with torch.no_grad():
            # for U in ctx.saved_tensors:
            #     print(U.shape)
            # print(ctx.d)
            d = ctx.d
            factors = ctx.saved_tensors[1:2*d+1]
            ndims = len(factors)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = (tt_shape[:d])
            tt_shape_col = (tt_shape[d:])

    

            
            B = len(dy.shape) - 1
            # B = 2
            dy = torch.flatten(dy,start_dim=0,end_dim=B-1)
            # if len(dy.shape)==3:
            #     dy = torch.flatten(dy,start_dim=0,end_dim=1)


            matrix = ctx.saved_tensors[0]
            T1 = ctx.saved_tensors[2*d+1]
            left = ctx.saved_tensors[2*d+2:2*d+2+d]
            right =  ctx.saved_tensors[2*d+2+d:]

            
            dy = dy.reshape(-1,*tt_shape_col)
            
            
            U1 = torch.tensordot(dy,right[-1],dims=[list(range(1,d+1)),list(range(1,d+1))])
            dx = torch.tensordot(U1,left[-1],dims=[[-1],[-1]])
            
            T2 = torch.tensordot(T1,dy,dims=[[0],[0]])
            right_grads = []
            
            grad = torch.tensordot(right[d-2],T2,dims=[list(range(0,d)),list(range(0,d))])[:,:,None]
            right_grads.append(grad)
            
            U = torch.squeeze(factors[-1])
            T2 = torch.tensordot(T2,U,[[-1],[1]])
            left_temp = right[d-3]
            grad = torch.tensordot(left_temp,T2,dims=[list(range(0,d-1)),list(range(0,d-1))])
            right_grads.append(grad)
            
            for i in range(2,d-1):
                U = factors[-(i)]
                left_temp = right[d-2-i]
                T2 = torch.tensordot(T2,U,[[-2,-1],[1,2]])
                grad = torch.tensordot(left_temp,T2,dims=[list(range(0,d-i),list(range(0,d-i)))])
                right_grads.append(grad)

            U = factors[-(d-1)]
            grad = torch.tensordot(T2,U,[[-2,-1],[1,2]])
            right_grads.append(grad)
            
            
            
            left_grads = []
            U2 = torch.tensordot(U1,matrix,[[0],[0]]).reshape(-1,*tt_shape_row).movedim(0,-1)
            
            grad = torch.tensordot(left[d-2],U2,dims=[list(range(0,d-1)),list(range(0,d-1))])
            left_grads.append(grad)
            
            U = torch.squeeze(factors[d-1])
            U2 = torch.tensordot(U2,U,dims=[[-2,-1],[1,2]])
            left_temp = left[d-3]
            grad = torch.tensordot(left_temp,U2,dims=[list(range(0,d-2)),list(range(0,d-2))])
            left_grads.append(grad)
            
            for i in range(2,d-1):
                U = torch.squeeze(factors[d-i])
                U2 = torch.tensordot(U2,U,dims=[[-2,-1],[1,2]])
                left_temp = left[d-2-i]
                grad = torch.tensordot(left_temp,U2,dims=[list(range(0,d-i-1)),list(range(0,d-i-1))])
                left_grads.append(grad)
                
            U = factors[1]
            grad = torch.tensordot(U2,U,dims=[[-2,-1],[1,2]])
            left_grads.append(grad)
            
            all_grads = left_grads[::-1] + right_grads[::-1]
            all_grads[0] = all_grads[0][None,:,:]




        return dx.reshape(*ctx.input_shape), *(all_grads)
    