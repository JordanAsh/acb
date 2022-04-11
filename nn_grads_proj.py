import torch
import math
import pdb
from torch.nn import functional as F
import types

def del_attr(obj, names):
    if len(names) == 1:
        if names[0].isnumeric():
            idx = int(names[0])
            obj[idx] = None
        else:
            delattr(obj, names[0])
    else:
        if names[0].isnumeric():
            idx = int(names[0])
            del_attr(obj[idx], names[1:])
        else:
            del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        if names[0].isnumeric():
            idx = int(names[0])
            obj[idx] = val
        else:
            setattr(obj, names[0], val)
    else:
        if names[0].isnumeric():
            idx = int(names[0])
            set_attr(obj[idx], names[1:], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)

def get_attr(obj, names):
    if len(names) == 1:
        if names[0].isnumeric():
            idx = int(names[0])
            return obj[idx]
        else:
            return getattr(obj, names[0])
    else:
        if names[0].isnumeric():
            idx = int(names[0])
            return get_attr(obj[idx], names[1:])
        else:
            return get_attr(getattr(obj, names[0]), names[1:])

def reparam(net, params, fields):
    for i,name in enumerate(fields):
        del_attr(net, name.split("."))
        set_attr(net, name.split("."), params[i])

    net.parameters = types.MethodType(lambda self: params, type(net))
    net.named_parameters = types.MethodType(lambda self: zip(fields, params), type(net))


def get_nn_grads_fn(net):
    fields = [name for name, w in net.named_parameters()]
        
    def grads_fn(x, n_outputs=None, badge=False, projections=None, offset=None, y=None):
        def f(*weights):
            if projections is None:
                reparam(net, weights, fields)
            else:
                reparam(net, [param + (weights[0] @ P.reshape(P.shape[0],-1)).reshape(P.shape[1:]) for param,P in zip(params,projections)], fields)

            nSamps = len(x)
            if badge:
                output = net(x)
                probs = F.softmax(output, 1)
                if y is None: labels = torch.argmax(output, 1).detach().long()
                elif len(y) == 0:
                    labels = torch.stack([torch.distributions.Categorical(probs[i]).sample() for i in range(nSamps)])
                else: labels = torch.Tensor(y).cuda().long()
                probs = torch.stack([probs[i, labels[i]] for i in range(nSamps)]).detach()
                ce = F.cross_entropy(output, labels, reduction='none')
                return ce * torch.sqrt(probs)
            else:

                output = net(x.repeat(n_outputs, 1, 1, 1))
                labels = torch.cat([torch.zeros(nSamps) + i for i in range(n_outputs)], 0).long().cuda()
                probs = F.softmax(output, 1)
                probs = torch.stack([probs[i, labels[i]] for i in range(len(labels))]).detach()
                ce = F.cross_entropy(output, labels, reduction='none')
                return ce * torch.sqrt(probs)
        
        params = tuple(get_attr(net, name.split(".")) for name in fields)
        if projections is None:
            jacobians = torch.autograd.functional.jacobian(f, params)
        else:
            jacobians = torch.autograd.functional.jacobian(f, (torch.zeros(projections[0].shape[0]).cuda(),))
        flat_jacobians = []
        
        if badge: out_size = len(x)
        else: out_size = len(x) * n_outputs
        for J in jacobians:
            J_ba = J.view(out_size, -1)
            flat_jacobians.append(J.view(J_ba.shape[0],-1))
        grads = torch.hstack(flat_jacobians)
        if badge: grads = grads.view(len(x), -1)
        else: grads = grads.view(n_outputs * len(x), -1)
        if offset is not None: grads = torch.cos(grads + offset)
        return grads
       

    return grads_fn


#net = torch.nn.Sequential( torch.nn.Linear(5,2), torch.nn.Linear(2,2) )
#x = torch.randn(3,5)

#net_dim = sum(p.numel() for p in net.parameters())
#proj_dim = 5

## orthogonal projection
#proj = torch.zeros(proj_dim, net_dim)
#torch.nn.init.orthogonal_(proj)
## RFF
#gamma = 1.
#proj = math.sqrt(2 * gamma) * torch.randn(proj_dim, net_dim)
#random_offset = torch.rand(proj_dim) * 2 * math.pi
#projections = []
#param_pos = 0
#for p in net.parameters():
#    param_dim = p.numel()
#    proj_shape = [proj_dim] + list(p.shape)
#    projections.append(proj[:,param_pos:param_pos+param_dim].reshape(proj_shape))
#    param_pos += param_dim
#grads_fn = get_nn_grads_fn(net)
#grads = grads_fn(x,2,badge=True,projections=projections)
#rff = torch.cos(grads + random_offset)
