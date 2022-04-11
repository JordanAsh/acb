from nn_grads import *
import torch
import pdb

net = torch.nn.Sequential(torch.nn.Linear(5, 3, bias=False), torch.nn.Linear(3, 2, bias=False)).cuda()
#net = torch.nn.Linear(5, 3, bias=False).cuda()

grads_fn = get_nn_grads_fn(net) # call once
x = torch.randn(10, 5).cuda()
grads = grads_fn(x, n_outputs=2, badge=False)


