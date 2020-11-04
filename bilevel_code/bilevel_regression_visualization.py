import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import numpy

'''
The generalization gap is estimated by the gap between the training error and the test error.

In particular, we save the model which achieves the smallest training error and test it over the test tasks and compute the error as the test error. Hence, the gap is estimated by abs(model['val_loss'][0]-min(model['train_loss'])).
'''

path='./sq_regression_bilevel/'
path_loo='./loo_regression_bilevel/'
font = {'size': 20}
matplotlib.rc('font', **font)

task_num=1000
print_task_num=True
print_loovssq_shot=True
print_high=True

if print_task_num:
    l_val=[]
    l_train=[]
    l_gap=[]
    query=1
    shot=5
    name='gap'
    run_time=0
    num_task=[10,33,100,333,1000,3333,10000]
    log_num_task=[np.log10(i) for i in num_task]
    for i in num_task:
        model=torch.load(path+'bilevel_sq_task'+str(task_num)+'_query'+str(query)+'_shot'+str(shot)+'_'+str(run_time)+'/trlog')
        l_val.append(model['val_loss'][0])
        l_train.append(min(model['train_loss']))
        l_gap.append(abs(model['val_loss'][0]-min(model['train_loss'])))

    plt.plot(log_num_task, l_val, linewidth=5,label='$\mathcal{R}$')
    plt.plot(log_num_task, l_val, 'v', ms=14)
    plt.plot(log_num_task, l_train, linewidth=5, label='$\hat{\mathcal{R}}_{s/q}$')
    plt.plot(log_num_task, l_train, 'v', ms=14)
    plt.plot(log_num_task,l_gap,linewidth=6, linestyle=':',color='black',label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$')
    plt.plot(log_num_task,l_gap,'v',ms=16)
    plt.ylabel('Error', fontsize=24)
    #plt.title('m={}, q={}'.format(shot, query))

    plt.xlabel('$\log$# of tasks',fontsize=24)
    plt.grid(True)

    if shot==5:
        plt.tight_layout()
        plt.legend()
    if shot==1:
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('small')
        plt.tight_layout()
        if query==1:
            plt.legend(bbox_to_anchor=(0.57, 0.6),prop=fontP)
        if query==15:
            plt.legend(bbox_to_anchor=(0.57, 0.57),prop=fontP)
    #plt.savefig('./ours_{}_query{}_shot{}.pdf'.format(name,query,shot),dpi=200)
    plt.show()


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
        for run_time in range(10):
            model=torch.load(path+'bilevel_sq_task'+str(task_num)+'_query15_shot'+str(shot)+'_'+str(run_time)+'/trlog')
            gap_list.append(abs(model['val_loss'][0]-min(model['train_loss'])))
        gap_sq.append(np.mean(np.array(gap_list)))
        gap_sq_upper.append(np.mean(np.array(gap_list))+np.std(np.array(gap_list)))
        gap_sq_lower.append(np.mean(np.array(gap_list))-np.std(np.array(gap_list)))
    for shot in shot_list:
        gap_list=[]
        for run_time in range(10):
            model=torch.load(path_loo+'bilevel_loo_task'+str(task_num)+'_shot'+str(shot)+'_'+str(run_time)+'/trlog')
            gap_list.append(abs(model['val_loss'][0]-min(model['train_loss'])))
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
    #plt.savefig('./ours_gap_loo_vs_sq_shot.pdf')
    plt.show()


if print_high:
    gap_sq=[]
    gap_sq_upper=[]
    gap_sq_lower=[]
    shot_list=[2,20,200,2000,20000]
    for shot in shot_list:
        gap_list=[]
        for run_time in range(10):
            model=torch.load(path+'bilevel_sq_task'+str(task_num)+'_query15_shot'+str(shot)+'_'+str(run_time)+'/trlog')
            gap_list.append((model['val_loss'][0]-min(model['train_loss'])))
        gap_sq.append(np.mean(np.array(gap_list)))
        gap_sq_upper.append(np.mean(np.array(gap_list))+np.std(np.array(gap_list)))
        gap_sq_lower.append(np.mean(np.array(gap_list))-np.std(np.array(gap_list)))
    
    shot=[1,2,3,4,5]
    plt.fill_between(shot, gap_sq_upper,gap_sq_lower,alpha=0.3)
    plt.plot(shot, gap_sq, linewidth=5, label='$\mathcal{R}-\hat{\mathcal{R}}_{s/q}$')
    plt.plot(shot, gap_sq, 'v', ms=14)
    plt.ylabel('Error gap', fontsize=24)
    plt.xlabel('# of shots', fontsize=24)
    plt.xticks(shot,shot_list)
    plt.grid(True)
    plt.ylim(-0.1,0.2)
    plt.tight_layout()
    plt.legend()
    #plt.savefig('./ours_gap.pdf')
    plt.show()
    


