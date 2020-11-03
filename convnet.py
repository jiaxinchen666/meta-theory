import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
import math
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def conv_block_same(in_channels, out_channels):
    return nn.Sequential(
        Conv2dSame(in_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ZeroPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self,dim,way):
        super(Classifier,self).__init__()
        self.dim=dim
        self.way=way
        self.fc=nn.Linear(self.dim,self.way)
    def forward(self,x):
        return self.fc(x)

class Classifier_mlp2(nn.Module):
    def __init__(self,dim,way):
        super(Classifier_mlp2,self).__init__()
        self.dim=dim
        self.way=way
        self.fc=nn.Linear(self.dim,600)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(600,self.way)
    def forward(self,x):
        return self.fc1(self.relu(self.fc(x)))

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv3 = nn.Sequential(
            Conv2dSame(self.in_channels, self.out_channels, 3),
            nn.BatchNorm2d(self.out_channels, eps=0.001, momentum=0.95, affine=True),
            nn.ReLU(),
            Conv2dSame(self.out_channels, self.out_channels, 3),
            nn.BatchNorm2d(self.out_channels, eps=0.001, momentum=0.95, affine=True),
            nn.ReLU(),
            Conv2dSame(self.out_channels, self.out_channels, 3),
            nn.BatchNorm2d(self.out_channels, eps=0.001, momentum=0.95, affine=True)
        )
        self.conv1 = nn.Sequential(
            Conv2dSame(self.in_channels, self.out_channels, 1),
            nn.BatchNorm2d(self.out_channels, eps=0.001, momentum=0.95, affine=True)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def swish(self, x):
        return x * self.sigmoid(x)

    def forward(self, x):
        out = self.conv3(x)
        return self.relu(out + self.conv1(x))

class Res_block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3)
        )
        self.conv1 = nn.Sequential(
            Conv2dSame(self.in_channels, self.out_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def swish(self, x):
        return x * self.sigmoid(x)

    def forward(self, x):
        out = self.conv3(x)
        return self.relu(out + self.conv1(x))


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Res_block(3, 64)
        self.layer2 = Res_block(64, 128)
        self.layer3 = Res_block(128, 256)
        self.layer4 = Res_block(256, 512)
        self.encoder = nn.Sequential(
            self.layer1,
            nn.MaxPool2d(2, 2),
            self.layer2,
            nn.MaxPool2d(2, 2),
            self.layer3,
            nn.MaxPool2d(2, 2, padding=1),
            self.layer4,
            nn.AvgPool2d(11, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Convnet_classifier(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, way=5):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600
        self.linear=nn.Linear(1600,way)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        return self.linear(x)

class Convnet_same(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block_same(x_dim, hid_dim),
            conv_block_same(hid_dim, hid_dim),
            conv_block_same(hid_dim, hid_dim),
            conv_block_same(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class generator(nn.Module):
    def __init__(self, emb, way):
        super(generator, self).__init__()
        self.emb=emb
        self.way = way
        self.fc1 = nn.Linear(self.emb, 1600)
        self.fc2 = nn.Linear(1600, self.emb*self.way+self.way)

        self.relu = nn.ReLU()

    '''def reparameterize(self, mean, var):
        if self.training:
            eps = Variable(var.data.new(var.size()).normal_())
            return eps.mul(var).add(mean)
        else:
            return mean'''

    '''def encoder(self, x, output):
        rep = self.relu(self.fc1(x))
        logmean, logvar = torch.split(self.fc4(rep), output, dim=-1)
        mean = logmean.exp()
        var = logvar.exp()
        var = torch.ones(logvar.size()).cuda() * 1.0
        return'''

    def forward(self, x):
        W,b=torch.split(self.fc2(self.relu(self.fc1(x))),self.emb*self.way,dim=-1)
        return W,b

class Regression_meta(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Regression_meta, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.act = nn.LeakyReLU()
    def forward(self,x):
        #return self.act(self.fc1(x))
        return self.act(self.fc2(self.act(self.fc1(x))))

class Regression(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Regression, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act=nn.LeakyReLU()
        self.fc2=nn.Linear(self.hidden_dim,output_dim)
    def forward(self,x):
        #return self.fc2(x)
        return self.fc2(self.act(self.fc1(x)))

class Regression_maml(nn.Module):
    def __init__(self,hidden_dim):
        super(Regression_maml,self).__init__()
        self.hidden_dim=hidden_dim
        self.fc1=Linear_fw(1,self.hidden_dim)
        #self.fc1.bias.data.fill_(0)
        self.fc2=Linear_fw(self.hidden_dim,self.hidden_dim)
        #self.fc2.bias.data.fill_(0)
        self.fc3=Linear_fw(self.hidden_dim,1)
        #self.fc2.bias.data.fill_(0)
        self.model=nn.Sequential(
        self.fc1,
        nn.ReLU(),
        self.fc2,
        nn.ReLU(),
        self.fc3
        )

    def forward(self,x):
        return self.model(x)

class MLP_classifier(nn.Module):
    def __init__(self,dims):
        super(MLP_classifier,self).__init__()
        self.dims=dims
        self.fc1=nn.Linear(self.dims,5)
        self.fc2=nn.Linear(400,20)
        self.relu=nn.ReLU()
    def forward(self,x):
        return self.fc1(x)
        #return self.fc2(self.relu(self.fc1(x)))


class GlobalDiscriminator(nn.Module):
    def __init__(self, dims):
        super(GlobalDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # nn.Linear(int(np.prod(img_shape)), 512),
            nn.Linear(dims, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

#    def forward(self, in_feature, out_feature):
#        feature = torch.cat((in_feature, out_feature), dim=0)
#        validity = self.model(feature)
    def forward(self, feature):
        return self.model(feature)

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.Linear):
        L.weight.data.normal_(0,1.0)
