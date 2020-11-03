import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class Classifier(nn.Module):
    def __init__(self, dim, way):
        super(Classifier, self).__init__()
        self.dim = dim
        self.way = way
        self.fc = nn.Linear(self.dim, self.way)

    def forward(self, x):
        return self.fc(x)


class Ours(MetaTemplate):
    def __init__(self, model_func, test_n_support, n_way, n_support):
        super(Ours, self).__init__(model_func, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_step = 5
        self.test_step = 5
        self.train_lr = 0.01
        self.n_task = 4
        self.test_n_shot = test_n_support
        self.train_n_shot = n_support

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def set_forward(self, x, step, is_feature=False):
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support,
                                                                     *x.size()[2:])  # support data
        x_b_i = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,
                                                                     *x.size()[2:])  # query data
        y_a_i = Variable(
            torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).long()).cuda()  # label for support data

        z_a_i = self.forward(x_a_i)
        z_b_i = self.forward(x_b_i)

        classifier_innertask = Classifier(z_a_i.shape[-1], self.n_way).cuda()
        optimizer_innertask = torch.optim.Adam(classifier_innertask.parameters(), lr=self.train_lr)

        list_support = []
        list_acc = []
        for i in range(step):
            list_support.append(z_a_i)

        for param in self.feature.parameters():
            param.requires_grad = False

        for data_support in list_support:
            logits_support = classifier_innertask(data_support)
            loss_support = self.loss_fn(logits_support, y_a_i)
            # classifier_innertask.zero_grad()
            optimizer_innertask.zero_grad()
            loss_support.backward(retain_graph=True)
            optimizer_innertask.step()
            list_acc.append(loss_support.item())

        classifier_innertask.eval()
        classifier_innertask.zero_grad()
        for param in classifier_innertask.parameters():
            param.requires_grad = False
        for param in self.feature.parameters():
            param.requires_grad = True

        scores = classifier_innertask(z_b_i)
        return scores, list_acc

    def correct(self, x, step):
        scores, list_acc = self.set_forward(x, step)
        # scores = self.set_forward(x, step)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), list_acc

    def set_forward_loss(self, x, step):
        scores, list_acc = self.set_forward(x, step, is_feature=False)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss, list_acc

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()
        self.n_support = self.train_n_shot
        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            self.n_way = x.size(0)
            loss, list_acc = self.set_forward_loss(x, step=self.train_step)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                # for param in self.feature.parameters():
                #     param.requires_grad = True
                # print("outer loop and grad ture")
                loss_q = torch.stack(loss_all).sum(0)
                optimizer.zero_grad()

                loss_q.backward()
                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
                # print(list_acc)

    def test_loop(self, test_loader, return_std=False):  # overwrite parent function
        correct = 0
        count = 0
        acc_all = []
        self.n_support = self.test_n_shot

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_way = x.size(0)
            self.n_query = x.size(1) - self.n_support
            correct_this, count_this, list_acc = self.correct(x, step=self.test_step)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        # print(list_acc)
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean