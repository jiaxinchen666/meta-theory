import argparse
import os.path as osp
import math
import torch
import torch.nn.functional as F
#import matplotlib
import numpy
import numpy as np
#import matplotlib.pyplot as plt
import random
from backbone import Regression_meta, Regression
from utils import pprint, set_gpu, ensure_path, Averager, Timer

for run_time in range(1,10):
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-epoch', type=int, default=200)
        parser.add_argument('--train_shot', type=int,default=2)
        parser.add_argument('--test_shot', type=int,default=2)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--lr', default=0.01)
        parser.add_argument('--train_step', default=5)
        parser.add_argument('--test_step', default=5)
        parser.add_argument('--hdim',default=40)
        parser.add_argument('--n_task', type=int, default=1000)

        args = parser.parse_args()
    save_path='./loo_regression_bilevel/bilevel_loo_task'+str(args.n_task)+'_shot'+str(args.train_shot)+'_'+str(run_time)
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(save_path)

    model = Regression_meta(1,args.hdim,args.hdim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 100.0 # set a large number as the initialization

    timer = Timer()

    #form the dataset
    ampl_tr_list = np.random.uniform(-5, 5, args.n_task)
    phase_tr_list = np.random.uniform(0, 1, args.n_task) * math.pi
    tr_shot_list=[]
    tr_shot_y_list=[]
    task_tr_list=[]
    for amplitude,phase in zip(ampl_tr_list,phase_tr_list):
        amplitude=torch.tensor([amplitude]).cuda()
        phase = torch.tensor([phase]).cuda()
        x_shot = 10.0*(torch.rand(args.train_shot) - 0.5).cuda()  # sample K shots from [-5.0, 5.0]
        y_shot = amplitude * (torch.sin(x_shot) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_shot))
        tr_shot_list.append(x_shot)
        tr_shot_y_list.append(y_shot)
        task_tr_list.append([amplitude, phase])

    ampl_val_list=np.random.uniform(-5,5,1000)
    phase_val_list=np.random.uniform(0,1,1000)*math.pi
    val_shot_list = []
    val_shot_y_list = []
    val_query_list = []
    val_query_y_list = []
    task_val_list=[]
    for ampl,phase in zip(ampl_val_list,phase_val_list):
        ampl = torch.tensor([ampl]).cuda()
        phase = torch.tensor([phase]).cuda()
        x_shot = (torch.rand(args.test_shot) - 0.5).cuda() * 10.0  # sample K shots from [-5.0, 5.0]
        y_shot = ampl * (torch.sin(x_shot) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_shot))
        x_query = (torch.rand(20) - 0.5).cuda() * 10.0
        y_query = ampl * (torch.sin(x_query) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_query))
        val_shot_list.append(x_shot)
        val_shot_y_list.append(y_shot)
        val_query_list.append(x_query)
        val_query_y_list.append(y_query)
        task_val_list.append([ampl,phase])

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        tl=Averager()

        for step in range(args.n_task):
            shot_x = tr_shot_list[step].cuda().reshape(-1, 1)
            shot_y = tr_shot_y_list[step].cuda().reshape(-1, 1)
            loss_q=0
            for j in range(args.train_shot):
                x_shot=shot_x[torch.arange(shot_x.size(0))!=j]
                y_shot=shot_y[torch.arange(shot_y.size(0))!=j]
                x_query=shot_x[j]
                y_query=shot_y[j]
                x_shot = model(x_shot)
                x_query=model(x_query)
                regressor = Regression(args.hdim, 1).cuda()
                optimizer_innertask = torch.optim.Adam(regressor.parameters(), lr=args.lr)

                list_support=[]

                for i in range(args.train_step):
                    list_support.append(x_shot)

                list_acc=[]

                for data_support in list_support:
                    pred_y_shot=regressor(data_support)
                    loss_support=F.mse_loss(pred_y_shot.squeeze(),y_shot.squeeze())
                    optimizer_innertask.zero_grad()
                    loss_support.backward(retain_graph=True)
                    optimizer_innertask.step()
                    list_acc.append(loss_support.item())

                regressor.eval()
                pred_y_query = regressor(x_query)
                loss = F.mse_loss(pred_y_query.squeeze(), y_query.squeeze())
                loss_q=loss_q+loss
                tl.add(loss.item())

            optimizer.zero_grad()
            loss_q.backward(retain_graph=True)
            optimizer.step()

            loss = None
        tl = tl.item()

        print('epoch {}, train, loss={:.4f}'.format(epoch, tl))
        if tl < trlog['min_loss']:
            trlog['min_loss'] = tl
            save_model('min-loss')

        trlog['train_loss'].append(tl)

        torch.save(trlog, osp.join(save_path, 'trlog'))
        save_model('epoch-last')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

    model.eval()
    vl = Averager()

    model.load_state_dict(torch.load(save_path+'/min-loss.pth'))

    for step in range(1000):
        x_shot = val_shot_list[step]
        y_shot = val_shot_y_list[step]
        x_query = val_query_list[step]
        y_query = val_query_y_list[step]
        x_shot = model(x_shot.cuda().reshape(-1, 1))
        x_query = model(x_query.cuda().reshape(-1, 1))
        regressor = Regression(args.hdim, 1).cuda()
        optimizer_innertask = torch.optim.Adam(regressor.parameters(), lr=args.lr)

        list_support = []

        for i in range(args.test_step):
            list_support.append(x_shot)

        for data_support in list_support:
            pred_y_shot = regressor(data_support)
            loss_support = F.mse_loss(pred_y_shot.squeeze(), y_shot.cuda())
            optimizer_innertask.zero_grad()
            loss_support.backward(retain_graph=True)
            optimizer_innertask.step()

        regressor.eval()
        pred_y_query = regressor(x_query)
        loss = F.mse_loss(pred_y_query.squeeze(), y_query.cuda())
        vl.add(loss.item())

        loss = None
    vl = vl.item()
    trlog['val_loss'].append(vl)
    torch.save(trlog, osp.join(save_path, 'trlog'))
    print('epoch {}, test, loss={:.4f}'.format(epoch, vl))





