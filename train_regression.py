import argparse
import os.path as osp
import math
import torch
import torch.nn.functional as F
import matplotlib
import numpy
import numpy as np
import matplotlib.pyplot as plt
import random
from convnet import Regression_meta,Regression
from utils import pprint, set_gpu, ensure_path, Averager, Timer

for query_num in [1,15]:
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-epoch', type=int, default=200)
        parser.add_argument('--train_shot', type=int, default=1)
        parser.add_argument('--test_shot', type=int, default=1)
        parser.add_argument('--query', type=int, default=query_num)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--lr', default=0.01)
        parser.add_argument('--train_step', default=5)
        parser.add_argument('--test_step', default=5)
        parser.add_argument('--hdim',default=40)
        parser.add_argument('--n_task',default=1000)

        args = parser.parse_args()
    save_path='./ours_task'+str(args.n_task)+'_query'+str(args.query)+'_shot'+str(args.train_shot)
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
    trlog['max_loss'] = 100.0

    timer = Timer()

    #form the dataset
    ampl_tr_list = np.random.uniform(-5, 5, args.n_task)
    phase_tr_list = np.random.uniform(0, 1, args.n_task) * math.pi
    tr_shot_list=[]
    tr_shot_y_list=[]
    tr_query_list = []
    tr_query_y_list = []
    task_tr_list=[]
    for amplitude,phase in zip(ampl_tr_list,phase_tr_list):
        amplitude=torch.tensor([amplitude]).cuda()
        phase = torch.tensor([phase]).cuda()
        x_shot = 10.0*(torch.rand(args.train_shot) - 0.5).cuda()  # sample K shots from [-5.0, 5.0]
        y_shot = amplitude * (torch.sin(x_shot) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_shot))
        x_query = (torch.rand(args.query) - 0.5).cuda()*10.0
        y_query = amplitude * (torch.sin(x_query) * torch.cos(phase) + torch.sin(phase) * torch.cos(x_query))
        tr_shot_list.append(x_shot)
        tr_shot_y_list.append(y_shot)
        tr_query_list.append(x_query)
        tr_query_y_list.append(y_query)
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
            x_shot = tr_shot_list[step]
            y_shot = tr_shot_y_list[step]
            x_query = tr_query_list[step]
            y_query = tr_query_y_list[step]
            x_shot = model(x_shot.cuda().reshape(-1, 1))
            x_query = model(x_query.cuda().reshape(-1, 1))
            regressor = Regression(args.hdim, 1).cuda()
            optimizer_innertask = torch.optim.Adam(regressor.parameters(), lr=args.lr)

            list_support=[]

            for i in range(args.train_step):
                list_support.append(x_shot)

            list_acc=[]

            for data_support in list_support:
                pred_y_shot=regressor(data_support)
                loss_support=F.mse_loss(pred_y_shot.squeeze(),y_shot.cuda())
                optimizer_innertask.zero_grad()
                loss_support.backward(retain_graph=True)
                optimizer_innertask.step()
                list_acc.append(loss_support.item())

            regressor.eval()
            pred_y_query = regressor(x_query)
            loss = F.mse_loss(pred_y_query.squeeze(), y_query.cuda())
            tl.add(loss.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss = None
        tl = tl.item()
        print('epoch {}, train, loss={:.4f}'.format(epoch, tl))

        if tl < trlog['max_loss']:
            trlog['max_loss'] = tl
            save_model('max-loss')

        trlog['train_loss'].append(tl)
        torch.save(trlog, osp.join(save_path, 'trlog'))
        save_model('epoch-last')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

    model.load_state_dict(torch.load(save_path + '/max-loss.pth'))

    model.eval()
    vl = Averager()

    for step in range(600):
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
    print('epoch {}, val, loss={:.4f}'.format(epoch, vl))

