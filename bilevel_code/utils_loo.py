import os
import shutil
import time
import pprint
import math
import numpy as np
import numpy
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-path',default='save_way10_norm1000')
    parser.add_argument('--train_way',default=5)
    parser.add_argument('--shot',default=10)

    args = parser.parse_args()

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def dot_metric_normalize_loo(a, b,B):
    a=torch.div(a, torch.norm(a, dim=1).unsqueeze(1)).unsqueeze(1)
    b=torch.div(b, torch.norm(b,dim=2).unsqueeze(2)).transpose(1,2)
    logits=torch.bmm(a, b).squeeze()
    logits_scaled=logits*B
    return logits,logits_scaled

def dot_metric_loo(a, b):
    return torch.bmm(a.unsqueeze(1), b.transpose(1,2)).squeeze()

def dot_metric(a, b):
    return torch.mm(a, b.t())

def dot_metric_normalize(a,b,B):
    return B*torch.mm(torch.div(a,torch.norm(a,dim=1).unsqueeze(1)), torch.div(b,torch.norm(b,dim=1).unsqueeze(1)).t())

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def euclidean_metric_loo(a, b, way):
    a = a.unsqueeze(1).expand(-1, way, -1)
    #logits = -(((a - b)**2).sum(dim=2)).sqrt()
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def euclidean_metric_normalize_loo(a, b, way,B):
    a = torch.div(a, torch.norm(a, dim=1).unsqueeze(1))
    b = torch.div(b, torch.norm(b, dim=2).unsqueeze(2))
    a = a.unsqueeze(1).expand(-1, way, -1)
    logits = -((a - b)**2).sum(dim=2)
    logits_scaled=logits*B
    return logits,logits_scaled

def euclidean_metric_normalize(a, b, B):
    a=torch.div(a, torch.norm(a, dim=1).unsqueeze(1))
    b=torch.div(b, torch.norm(b, dim=1).unsqueeze(1))
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits=-((a - b)**2).sum(dim=2)
    logits_scaled = logits* (B/2)
    return logits,logits_scaled

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

'''def output(f):
    model = torch.load('./'+f+'/'+'trlog')
    return print(model['train_loss'][list.index(model['val_acc'], max(model['val_acc']))]), print(model['train_loss'][list.index(model['val_loss'], min(model['val_loss']))])
'''

def output(f):
    model=torch.load('./'+f+'/'+'trlog')
    return print(min(model['train_loss'])),print(model['train_loss'][list.index(model['train_acc'], max(model['train_acc']))])

class scaling(nn.Module):
    def __init__(self,margin):
        super(scaling,self).__init__()
        self.margin=margin
        return
    def forward(self,logits,label):
        mask=numpy.ones(logits.shape)
        for i in range(logits.shape[0]):
            mask[i][label[i]]=self.margin
        mask=torch.from_numpy(mask).float().cuda()
        return logits*mask

class sample(nn.Module):
    def __init__(self,mean,std):
        super(sample,self).__init__()
        self.mean=mean
        self.std=std
        return

    def forward(self,logits):
        B = torch.normal(mean=torch.tensor(self.mean), std=torch.tensor(self.std)).cuda()
        return logits*torch.abs(B)

class cross_entropy_margin(nn.Module):
    def __init__(self,margin):
        super(cross_entropy_margin,self).__init__()
        self.margin=margin
        return

    def forward(self, logits, label):
        loss=0
        for i in range(logits.shape[0]):
            loss -= torch.tensor(torch.div(torch.exp(logits[i][label[i]]*self.margin),
                     torch.sum(torch.exp(logits[i]*(1/self.margin)))-torch.exp(logits[i][label[i]]*(1/self.margin))
                                            +torch.exp(logits[i][label[i]]*self.margin)
                     ).log(),requires_grad=True)
        return loss/logits.shape[0]

class Bufferswitch(nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.register_buffer('mask',torch.ones(1))

    def set(self, mask):
        self.mask.data.copy_(mask)

    def get(self):
        return self.mask.data

def proto_compute(data,i,way,shot):  #10,5,1600
    temp=data.clone()
    dim = torch.ones([way, 1]).cuda()*shot
    temp[i//way,i%way]=torch.zeros([5])
    dim[i%way,0]=shot-1
    return temp.sum(dim=0).div(dim)

#print(average_acc)
#print(sum((np.array(accuracy_list)-average_acc)*(np.array(accuracy_list)-average_acc)))


'''gap=[]
for i in range(5):
    gap.append(output('benchmark/euclidean_1shot_benchmark_'+str(i)))
print(sum(gap)/5,gap)
output('benchmark/euclidean_benchmark_0')
output('benchmark/euclidean_benchmark_1')
#output('benchmark/euclidean_1shot_benchmark_1')

A=torch.tensor([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.],[4.,4.,4.],[5.,5.,5.],
                [6.,6.,6.],[7.,7.,7.],[8.,8.,8.],[9.,9.,9.],[10.,10.,10.]])
B=copy.deepcopy(A).reshape(2,5,3) #A is data, B is proto
protos=[proto_compute(B,i,5,2) for i in range(10)]
protos=torch.stack(protos)

#print(protos,protos.transpose(1,2))

print(A.unsqueeze(1).expand(10,5, -1).shape)
#b.unsqueeze(0).expand(n, m, -1)
#logits = -((a - b)**2).sum(dim=2)
A=torch.tensor([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.],[4.,4.,4.],[5.,5.,5.],
                [6.,6.,6.],[7.,7.,7.],[8.,8.,8.],[9.,9.,9.],[10.,10.,10.]])
B=copy.deepcopy(A).reshape(2,5,3) #A is data, B is proto
print(B,B.shape)
print(torch.div(B,torch.norm(B,dim=2).unsqueeze(2)))'''


