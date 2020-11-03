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

for eee in range(300):
    path = './regression/shot10_d40_tsstep10_trstep5'
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--shot', type=int, default=10)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--lr', default=0.01)
        parser.add_argument('--load', default=path + '/max-loss.pth')
        parser.add_argument('--step', default=15)
        args = parser.parse_args()

        model=Regression_meta(1,40,40).cuda()

        model.load_state_dict(torch.load(args.load))
        model.eval()

        ampl_plot=(torch.rand(1)-0.5).cuda()*10
        phase_plot=torch.rand(1).cuda()*math.pi
        #x_plot_shot = (torch.rand(args.shot) - 0.5).cuda() * 10.0  # sample K shots from [-5.0, 5.0]
        x_plot_shot = torch.rand(args.shot).cuda() * 5.0
        y_plot_shot = ampl_plot * (torch.sin(x_plot_shot) * torch.cos(phase_plot) + torch.sin(phase_plot) * torch.cos(x_plot_shot))
        x_plot_model=model(x_plot_shot.reshape(-1,1))

        regressor = Regression(40, 1).cuda()
        optimizer_innertask = torch.optim.Adam(regressor.parameters(), lr=args.lr)

        list_support = []

        for i in range(args.step):
            list_support.append(x_plot_model)

        for data_support in list_support:
            pred_y_shot = regressor(data_support)
            loss_support = F.mse_loss(pred_y_shot.squeeze(), y_plot_shot.cuda())
            optimizer_innertask.zero_grad()
            loss_support.backward(retain_graph=True)
            optimizer_innertask.step()

        regressor.eval()

        font = {'size': 20}
        matplotlib.rc('font', **font)
        x = numpy.arange(-5.0, 5.0, 0.01)
        x_plot = torch.from_numpy(x).float()
        true_y = ampl_plot * (torch.sin(x_plot.cuda()) * torch.cos(phase_plot) + torch.sin(phase_plot) * torch.cos(x_plot.cuda()))
        plt.plot(x, true_y.tolist(), label='Ground truth', linewidth=5, color='r')
        plt.plot(x_plot_shot.tolist(), y_plot_shot.tolist(), 'v', markersize =20, label='Training shots')
        pred_x = numpy.arange(-5.0, 5.0, 0.05)
        pred_x_plot = torch.from_numpy(pred_x).float()
        pred_y = regressor(model(pred_x_plot.cuda().reshape(-1, 1)))
        plt.plot(pred_x, pred_y.squeeze().tolist(), '--',label='Predicted',linewidth=5,color='g')
        plt.ylim(-7, 7)
        plt.xlim(-5, 5)
        plt.ylabel('y', fontsize=24)
        plt.xlabel('x', fontsize=24)
        plt.grid(True)
        if abs(ampl_plot)<0.5:
            plt.legend()
        plt.tight_layout()
        plt.savefig('/home/chenjiaxin/Dropbox/figure/'+str(eee)+'.pdf')
        plt.show()

