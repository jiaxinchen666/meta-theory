import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import numpy
font = {'size': 20}
matplotlib.rc('font', **font)

def get_val(model):
    return model['val_loss'][list.index(model['train_loss'], get_train(model))]
def get_train(model):
    return min(model['train_loss'])
def get_gap(model):
    return abs(get_train(model) - get_val(model))

print_task_num=True
print_loovssq_shot=True

if print_task_num:
    l_val=[]
    l_train=[]
    l_gap=[]
    query=1
    shot=5
    name='gap'
    num_task=[10,33,100,333,1000,3333,10000]
    log_num_task=[np.log10(i) for i in num_task]
    for i in num_task:
        model_sq = torch.load(
            './SQ_regression_bilevel/ours_task' + str(i) + '_query' + str(query) + '_shot' + str(shot) + '/trlog')
        if shot==1 and i==1000: #we directly save the 'min_train_loss' model and test it
            l_val.append(model_sq['val_loss'][0])
            l_train.append(get_train(model_sq))
            l_gap.append(abs(get_train(model_sq)-model_sq['val_loss'][0]))
        else:
            l_val.append(get_val(model_sq))
            l_train.append(get_train(model_sq))
            l_gap.append(get_gap(model_sq))

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
    plt.savefig('./figure/ours_{}_query{}_shot{}.pdf'.format(name,query,shot),dpi=200)
    plt.show()

if print_loovssq_shot:
    gap_sq=[]
    gap_loo=[]
    gap_sq_q1=[]
    shot_list=[2,3,5,7,9,10]
    for shot in shot_list:
        model_sq=torch.load('./SQ_regression_bilevel/ours_task'+str(1000)+'_query'+str(15)+'_shot'+str(shot)+'/trlog')
        model_sq_q1 = torch.load('./SQ_regression_bilevel/ours_task' + str(1000) + '_query' + str(1) + '_shot' + str(shot) + '/trlog')
        model_loo=torch.load('./LOO_regression_bilevel/looregression_task'+str(1000)+'_shot'+str(shot)+'/trlog')
        gap_sq.append(get_gap(model_sq))
        if shot==3: #for query=1 shot=3 we directly save the 'min_train_loss' model and test it.
            model_sq_q1=torch.load('./SQ_regression_bilevel/ours_task'+str(1000)+'_query'+str(1)+'_shot'+str(shot)+'/trlog')
            gap_sq_q1.append(abs(min(model_sq_q1['train_loss'])-model_sq_q1['val_loss'][0]))
        else:
            gap_sq_q1.append(get_gap(model_sq_q1))
        gap_loo.append(get_gap(model_loo))
    shot=shot_list
    plt.plot(shot, gap_sq, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$, q=15')
    plt.plot(shot, gap_sq, 'v', ms=14)
    plt.plot(shot, gap_sq_q1, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{s/q}|$, q=1')
    plt.plot(shot, gap_sq_q1, 'v', ms=14)
    plt.plot(shot, gap_loo, linewidth=5, label='$|\mathcal{R}-\hat{\mathcal{R}}_{loo}|$')
    plt.plot(shot, gap_loo, 'v', ms=14)
    plt.ylabel('Error gap', fontsize=24)
    plt.xlabel('# of shots', fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('./figure/ours_gap_loo_vs_sq_shot.pdf')
    plt.show()

