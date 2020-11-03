import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import numpy
font = {'size': 20}
matplotlib.rc('font', **font)

def get_train(model):
    return min(model['train_loss'])
    #return round(model['train_loss'][list.index(model['val_loss'], min(model['val_loss']))],6)
def get_val(model):
    return model['val_loss'][list.index(model['train_loss'], min(model['train_loss']))]
def get_gap(model):
    return abs(get_train(model) - get_val(model))

print_task_num=True
print_reg_loovssq_shot=True
print_reg_task_num=True

if print_task_num:
    l_val=[]
    l_train=[]
    l_gap=[]
    query=1
    shot=1
    name='gap'
    algorithm='protonet'
    tasks = [100,500,1500,2500,3500,4500,5500]
    for task_num in tasks:
        if algorithm=='maml':
            model = torch.load(
            './CloserLookFewShot/checkpoints/miniImagenet/Conv4_maml_approx_' + str(task_num) + '_5way_1shot/trlog')
        else:
            model = torch.load(
                './CloserLookFewShot/checkpoints/miniImagenet/Conv4_protonet_' + str(task_num) + '_5way_1shot/trlog')
        l_val.append(round(min(model['val_loss']), 4))
        l_train.append(round(model['train_loss'][list.index(model['val_loss'], min(model['val_loss']))], 4))
        l_gap.append(round(
            abs(min(model['val_loss']) - model['train_loss'][list.index(model['val_loss'], min(model['val_loss']))]),
            4))
    plt.plot(tasks, l_val, linewidth=5, label='$\mathcal{R}$')
    plt.plot(tasks, l_val, 'v', ms=14)
    plt.plot(tasks, l_train, linewidth=5, label='$\hat{\mathcal{R}}_{s/q}$')
    plt.plot(tasks, l_train, 'v', ms=14)
    plt.plot(tasks, l_gap, linewidth=6, linestyle=':', color='black',
		 label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$')
    plt.plot(tasks, l_gap, 'v', ms=16)
    plt.ylabel('Error', fontsize=24)

    plt.xlabel('# of tasks',fontsize=24)
    plt.grid(True)
    plt.xticks(numpy.arange(0,6000,2000))
    plt.tight_layout()
    plt.legend()
    plt.savefig('./figures/{}_{}_query{}_shot{}.pdf'.format(algorithm,name,query,shot),dpi=200)
    plt.show()

if print_reg_task_num:
    l_val=[]
    l_train=[]
    l_gap=[]
    query=1
    shot=1
    name='gap'
    algorithm='maml_reg'
    tasks_list = [10,33,100,333,1000,3333,10000]
    log_num_task=[np.log10(task) for task in tasks_list]
    for task_num in tasks_list:
        if algorithm=='maml_reg':
            model = torch.load(
            './CloserLookFewShot/checkpoints/regression/ReMaml_re_maml_' + str(task_num) + '_5shot/trlog')
        l_val.append(get_val(model))
        l_train.append(get_train(model))
        l_gap.append(get_gap(model))
    plt.plot(log_num_task, l_val, linewidth=5, label='$\mathcal{R}$')
    plt.plot(log_num_task, l_val, 'v', ms=14)
    plt.plot(log_num_task, l_train, linewidth=5, label='$\hat{\mathcal{R}}_{s/q}$')
    plt.plot(log_num_task, l_train, 'v', ms=14)
    plt.plot(log_num_task, l_gap, linewidth=6, linestyle=':', color='black',
                 label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$')
    plt.plot(log_num_task, l_gap, 'v', ms=16)
    plt.ylabel('Error', fontsize=24)

    plt.xlabel('$\log$# of tasks', fontsize=24)
    plt.grid(True)
    from matplotlib.font_manager import FontProperties

    fontP = FontProperties()
    fontP.set_size('small')
    # plt.yticks(numpy.arange(0,1,0.2))
    plt.tight_layout()
    plt.legend()
    #plt.legend(bbox_to_anchor=(0.57, 0.6), prop=fontP)
    plt.savefig('./figures/maml_{}_query{}_shot{}.pdf'.format(name, query, shot),
                dpi=200)
    plt.show()

if print_reg_loovssq_shot:
    gap_sq=[]
    gap_loo=[]
    gap_sq_q1=[]
    for shot_num in [2,3,5,7,9,10]:
        model_loo=torch.load('./CloserLookFewShot/checkpoints/regression_loo/ReMaml_re_maml_'+str(1000)+'_'+str(shot_num)+'shot/trlog')
        model_sq_q1 = torch.load('./CloserLookFewShot/checkpoints/regression/ReMaml_re_maml_' + str(1000) + '_1q_' + str(shot_num)+'shot/trlog')
        model_sq=torch.load('./CloserLookFewShot/checkpoints/regression/ReMaml_re_maml_' + str(1000) + '_' + str(shot_num)+'shot/trlog')
        gap_sq.append(get_gap(model_sq))
        gap_sq_q1.append(get_gap(model_sq_q1))
        gap_loo.append(get_gap(model_loo))
    shot=[2,3,5,7,9,10]
    plt.plot(shot, gap_sq, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$, q=15')
    plt.plot(shot, gap_sq, 'v', ms=14)
    plt.plot(shot, gap_sq_q1, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$, q=1')
    plt.plot(shot, gap_sq_q1, 'v', ms=14)
    plt.plot(shot, gap_loo, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{loo}|$')
    plt.plot(shot, gap_loo, 'v', ms=14)
    plt.ylabel('Error gap', fontsize=24)
    plt.xlabel('# of shots', fontsize=24)
    plt.grid(True)
    # plt.yticks(numpy.arange(0,1,0.2))
    plt.tight_layout()
    plt.legend()
    #plt.title('$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$ or $|\mathcal{R}-\hat{\mathcal{R}}_{loo}|$')
    plt.savefig('./figures/maml_gap_loovssq_shot.pdf')
    plt.show()








