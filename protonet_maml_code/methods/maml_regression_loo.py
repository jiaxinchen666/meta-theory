# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class MAML_regloo(MetaTemplate):
    def __init__(self, model_func, n_query, n_support, approx=False):
        super(MAML_regloo, self).__init__(model_func, n_query, n_support, change_way=False)

        self.loss_fn = nn.MSELoss()
        # self.regressor = backbone.Regression_maml(40)
        self.n_support = n_support
        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

    def forward(self, x):
        scores = self.feature.forward(x)
        return scores

    def set_forward(self, x_shot,x_query,y_shot, is_feature=False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x_shot = Variable(x_shot)
        y_shot = Variable(y_shot)
        x_query = Variable(x_query)

        fast_parameters = list(self.parameters())  # the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_shot)
            set_loss = self.loss_fn(scores, y_shot)
            grad = torch.autograd.grad(set_loss, fast_parameters,
                                       create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[
                        k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(
                    weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        scores = self.forward(x_query)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x_shot,y_shot,x_query,y_query):
        scores = self.set_forward(x_shot,x_query,y_shot, is_feature=False)
        # print(scores.size())
        y_b_i = Variable(y_query)
        # print(y_b_i.size())
        loss = self.loss_fn(scores, y_b_i)
        return loss

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()
        # train
        for i, x in enumerate(train_loader):
            # print(len(list(train_loader)))
            x_shot, y_shot, x_query, y_query=x
            x_shot = x_shot.view(self.n_support,-1)
            y_shot  = y_shot.view(self.n_support,-1)
            loss_loo=0
            for j in range(self.n_support):
                x_support = x_shot[torch.arange(x_shot.size(0)) != j]
                y_support = y_shot[torch.arange(y_shot.size(0)) != j]
                x_query = x_shot[j]
                y_query = y_shot[j]
                loss = self.set_forward_loss(x_support,y_support,x_query,y_query)
                loss_loo=loss_loo+loss
            avg_loss = avg_loss + loss_loo.item()/self.n_support
            loss_all.append(loss_loo)
            task_count += 1
            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
        print('Epoch {:d}| Loss {:f}'.format(epoch, avg_loss / float(len(list(train_loader)) + 1)))
        return avg_loss / float(len(list(train_loader)) + 1)

    def test_loop(self, test_loader, return_std=False):  # overwrite parrent function
        loss_all = []

        iter_num = len(list(test_loader))
        for i, (x) in enumerate(test_loader):
            x_shot, y_shot, x_query, y_query=x
            x_shot = x_shot.view(-1, 1)
            y_shot = y_shot.view(-1, 1)
            x_query = x_query.view(-1, 1)
            y_query = y_query.view(-1, 1)
            # assert self.n_way == x.size(0), "MAML do not support way change"
            loss=self.set_forward_loss(x_shot,y_shot,x_query,y_query)
            loss_all.append(loss.item())
        print('test:',sum(loss_all)/iter_num)
        return sum(loss_all)/iter_num