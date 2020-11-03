import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import math
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.regression_loo import MAML_regloo
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml_regression import MAML as ReMaml
from methods.maml import MAML
from methods.Ours import Ours
from io_utils import model_dict, parse_args, get_resume_file

def train(traindata, valdata, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        trainloss=model.train_loop(epoch, traindata,  optimizer ) #model are called by reference, no need to return
        trlog['train_loss'].append(trainloss)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        testloss= model.test_loop(valdata)
        trlog['val_loss'].append(testloss)
        '''if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)'''
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(1000)
    task_num=1000

    for shot_num in range(2,11,1):
        params = parse_args('train',task_num,shot_num)

        if params.dataset in ['regression','regression_loo']:
            ampl_tr_list = np.random.uniform(-5, 5, task_num)
            phase_tr_list = np.random.uniform(0, 1, task_num) * math.pi
            tr_shot_list = []
            tr_shot_y_list = []
            tr_query_list = []
            tr_query_y_list = []
            task_tr_list = []
            for amplitude, phase in zip(ampl_tr_list, phase_tr_list):
                amplitude = torch.tensor([amplitude]).cuda()
                phase = torch.tensor([phase]).cuda()
                x_shot = 10.0 * (torch.rand(params.n_shot) - 0.5).cuda()  # sample K shots from [-5.0, 5.0]
                y_shot = amplitude * (torch.sin(x_shot) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_shot))
                x_query = (torch.rand(params.n_query) - 0.5).cuda() * 10.0
                y_query = amplitude * (torch.sin(x_query) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_query))
                tr_shot_list.append(x_shot.reshape(-1, 1))
                tr_shot_y_list.append(y_shot)
                tr_query_list.append(x_query.reshape(-1, 1))
                tr_query_y_list.append(y_query)
                task_tr_list.append([amplitude, phase])

            ampl_val_list = np.random.uniform(-5, 5, 600)
            phase_val_list = np.random.uniform(0, 1, 600) * math.pi
            val_shot_list = []
            val_shot_y_list = []
            val_query_list = []
            val_query_y_list = []
            task_val_list = []
            for ampl, phase in zip(ampl_val_list, phase_val_list):
                ampl = torch.tensor([ampl]).cuda()
                phase = torch.tensor([phase]).cuda()
                x_shot = (torch.rand(params.n_shot) - 0.5).cuda() * 10.0  # sample K shots from [-5.0, 5.0]
                y_shot = ampl * (torch.sin(x_shot) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_shot))
                x_query = (torch.rand(15) - 0.5).cuda() * 10.0
                y_query = ampl * (torch.sin(x_query) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_query))
                val_shot_list.append(x_shot.reshape(-1, 1))
                val_shot_y_list.append(y_shot)
                val_query_list.append(x_query.reshape(-1, 1))
                val_query_y_list.append(y_query)
                task_val_list.append([ampl, phase])

        if params.dataset == 'cross':
            base_file = configs.data_dir['miniImagenet'] + 'all.json'
            val_file   = configs.data_dir['CUB'] + 'val.json'
        elif params.dataset == 'cross_char':
            base_file = configs.data_dir['omniglot'] + 'noLatin.json'
            val_file   = configs.data_dir['emnist'] + 'val.json'
        elif params.dataset in ['CUB','miniImagenet','omniglot']:
            base_file = configs.data_dir[params.dataset] + 'base.json'
            val_file   = configs.data_dir[params.dataset] + 'val.json'

        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        if params.dataset in ['omniglot', 'cross_char']:
            assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
            params.model = 'Conv4S'

        optimization = 'Adam'

        if params.stop_epoch == -1:
            if params.method in ['baseline', 'baseline++'] :
                if params.dataset in ['omniglot', 'cross_char']:
                    params.stop_epoch = 5
                elif params.dataset in ['CUB']:
                    params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
                elif params.dataset in ['miniImagenet', 'cross']:
                    params.stop_epoch = 400
                else:
                    params.stop_epoch = 400 #default
            else: #meta-learning methods
                if params.n_shot == 1:
                    params.stop_epoch = 300 #800
                elif params.n_shot == 5:
                    params.stop_epoch = 300
                else:
                    params.stop_epoch = 600 #default
        if params.dataset != 'regression':
            if params.method in ['baseline', 'baseline++'] :
                base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
                base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
                val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
                val_loader      = val_datamgr.get_data_loader( val_file, aug = False)

                if params.dataset == 'omniglot':
                    assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
                if params.dataset == 'cross_char':
                    assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

                if params.method == 'baseline':
                    model           = BaselineTrain( model_dict[params.model], params.num_classes)
                elif params.method == 'baseline++':
                    model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')

            elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx','Ours']:
                #n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

                n_query=params.n_query
                train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
                base_datamgr            = SetDataManager(image_size, n_query = n_query,n_episode=params.n_episodes,  **train_few_shot_params)
                base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

                test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot)
                val_datamgr             = SetDataManager(image_size, n_query = 15,n_episode=600, **test_few_shot_params)
                val_loader              = val_datamgr.get_data_loader( val_file, aug = False)
            elif params.method in ['protonet_loo','maml_loo']:
                n_query = params.n_query
                train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
                base_datamgr = SetDataManager(image_size, n_query=0, n_episode=params.n_episodes,
                                              **train_few_shot_params)
                base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

                test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
                val_datamgr = SetDataManager(image_size, n_query=15, n_episode=600, **test_few_shot_params)
                val_loader = val_datamgr.get_data_loader(val_file, aug=False)
                #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model= ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'protonet_loo':
            model= ProtoNet_loo( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':
            model= MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'Ours':
            model=Ours(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            train_few_shot_params = dict(n_support = params.n_shot,n_way=params.train_n_way)
            model= MAML(  model_dict[params.model], approx = (params.method == 'maml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
        elif params.method in ['maml_loo']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            train_few_shot_params    = dict(n_support = params.n_shot,n_query=params.n_query,n_way=params.train_n_way)
            model= MAML_loo(  model_dict[params.model], approx = (params.method == 'maml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
        elif params.method =="re_maml" and params.dataset == 'regression':
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            train_few_shot_params = dict(n_support=params.n_shot, n_query=params.n_query)
            model = ReMaml(model_dict[params.model], approx=True, **train_few_shot_params)

        elif params.method =="re_maml" and params.dataset == 'regression_loo':
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            train_few_shot_params = dict(n_support=params.n_shot, n_query=params.n_query)
            model = MAML_regloo(model_dict[params.model], approx=True, **train_few_shot_params)


        model = model.cuda()

        params.checkpoint_dir = '.%s/checkpoints/%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.model, params.method,params.n_episodes)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        if not params.method  in ['baseline', 'baseline++']:
            params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        if params.method == 'maml' or params.method == 'maml_approx' :
            stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update

        if params.resume:
            resume_file = get_resume_file(params.checkpoint_dir)
            if resume_file is not None:
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch']+1
                model.load_state_dict(tmp['state'])
        elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
            baseline_checkpoint_dir = '.%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
            if params.train_aug:
                baseline_checkpoint_dir += '_aug'
            warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
            tmp = torch.load(warmup_resume_file)
            if tmp is not None:
                state = tmp['state']
                state_keys = list(state.keys())
                for i, key in enumerate(state_keys):
                    if "feature." in key:
                        newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                        state[newkey] = state.pop(key)
                    else:
                        state.pop(key)
                model.feature.load_state_dict(state)
            else:
                raise ValueError('No warm_up file')

        if params.dataset in ['regression','regression_loo']:
            print('training regression ')
            traindata = list(zip(tr_shot_list, tr_shot_y_list, tr_query_list, tr_query_y_list))
            valdata = list(zip(val_shot_list, val_shot_y_list, val_query_list, val_query_y_list))
            # print(traindata)
            # print(valdata)
            model = train(traindata, valdata, model, optimization, start_epoch, stop_epoch, params)
        else:
            if params.gap:
                traindata=[x for i, (x, _) in enumerate(base_loader)]
                valdata=[x for i, (x, _) in enumerate(val_loader)]
                model = train(traindata, valdata,  model, optimization, start_epoch, stop_epoch, params)
            else:
                model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
