# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:31:48 2020
"""
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import numpy

'''
The generalization gap is estimated by the gap between the training error and the test error.

In particular, we save the model which achieves the smallest training error and test it over the test tasks and compute the error as the test error.

As the training procedure of MAML is unstable and the test variance is big, we compute the average of the last five gaps (see get_gap).
'''

path='./regression/'
path_loo='./regression_loo/'
font = {'size': 20}
matplotlib.rc('font', **font)

def get_val(model,i):
    return model['val_loss'][list.index(model['train_loss'], get_train(model,i))]
def get_train(model,i):
    return sorted(model['train_loss'])[i]
def get_gap(model):
    gap_list=[]
    for i in range(5):
        if abs((get_val(model,i)-get_train(model,i)))<1:
            gap_list.append(get_val(model,i)-get_train(model,i))
    return np.mean(np.array(gap_list))

task_num=1000
print_loovssq_shot=True
print_high=True

if print_loovssq_shot:
    gap_sq=[]
    gap_loo=[]
    gap_sq_upper=[]
    gap_sq_lower=[]
    gap_loo_upper=[]
    gap_loo_lower=[]
    shot_list=[3,4,5,6,7,8,9,10]
    for shot in shot_list:
        gap_list=[]
        for run_time in range(20):
            model=torch.load(path+'ReMaml_1000_'+str(shot)+'shot_15query'+'_'+str(run_time)+'/trlog')
            if get_gap(model)<1:
                gap_list.append(abs(get_gap(model)))
        gap_sq.append(np.mean(np.array(gap_list)))
        gap_sq_upper.append(np.mean(np.array(gap_list))+np.std(np.array(gap_list)))
        gap_sq_lower.append(np.mean(np.array(gap_list))-np.std(np.array(gap_list)))
    for shot in shot_list:
        gap_list=[]
        for run_time in range(20):
            model=torch.load(path_loo+'ReMaml_1000_'+str(shot)+'shot'+'_'+str(run_time)+'/trlog')
            if get_gap(model)<1:
                gap_list.append(abs(get_gap(model)))
        gap_loo.append(np.mean(np.array(gap_list)))
        gap_loo_upper.append(np.mean(np.array(gap_list))+np.std(np.array(gap_list)))
        gap_loo_lower.append(np.mean(np.array(gap_list))-np.std(np.array(gap_list)))

    shot=shot_list
    plt.plot(shot, gap_sq, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$')
    plt.plot(shot, gap_sq, 'v', ms=14, color='red')
    plt.fill_between(shot, gap_sq_upper,gap_sq_lower,alpha=0.3)
    plt.plot(shot, gap_loo, linewidth=5, color='orange', label='$|\mathcal{R}-\hat{\mathcal{R}}_{loo}|$')
    plt.plot(shot, gap_loo, 'v', ms=14,color='g')
    plt.fill_between(shot, gap_loo_upper,gap_loo_lower,color='orange',alpha=0.3)
    plt.ylabel('Error gap', fontsize=24)
    plt.xlabel('# of shots', fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    #plt.savefig('./maml_gap_loo_vs_sq_shot.pdf')
    plt.show()


if print_high:
    gap_sq=[]
    gap_sq_upper=[]
    gap_sq_lower=[]
    shot_list=[2,20,200,2000,20000]
    for shot in shot_list:
        gap_list=[]
        for run_time in range(20):
            model=torch.load(path+'ReMaml_1000_'+str(shot)+'shot_15query'+'_'+str(run_time)+'/trlog')
            if get_gap(model)<1:
                gap_list.append(abs(get_gap(model)))
        gap_sq.append(np.mean(np.array(gap_list)))
        gap_sq_upper.append(np.mean(np.array(gap_list))+np.std(np.array(gap_list)))
        gap_sq_lower.append(np.mean(np.array(gap_list))-np.std(np.array(gap_list)))
    
    shot=[1,2,3,4,5]
    plt.fill_between(shot, gap_sq_upper,gap_sq_lower,alpha=0.5)
    plt.plot(shot, gap_sq, linewidth=5, label='$\mathcal{R}-\hat{\mathcal{R}}_{s/q}$')
    plt.plot(shot, gap_sq, 'v', ms=14)
    plt.ylabel('Error gap', fontsize=24)
    plt.xlabel('# of shots', fontsize=24)
    plt.xticks(shot,shot_list)
    plt.grid(True)
    plt.ylim(-0.5,0.5)
    plt.tight_layout()
    plt.legend()
    #plt.savefig('./figure/ours_gap_loo_vs_sq_shot.pdf')
    plt.show()

