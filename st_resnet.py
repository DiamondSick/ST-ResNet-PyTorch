# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# %%
class ResUnit(nn.Module):
    def __init__(self,in_dim,nb_filter=64):
        super(ResUnit,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim,nb_filter,3,1,1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Conv2d(in_dim,nb_filter,3,1,1)
    def forward(self,x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

    
class ResSeq(nn.Module):
    def __init__(self,in_dim,nb_filter,nb_residual_unit):
        super(ResSeq,self).__init__()
        layers = []
        for i in range(nb_residual_unit):
            layers.append(ResUnit(in_dim,nb_filter))
        self.resLayer = nn.Sequential(*layers)
    def forward(self,x):
        return self.resLayer(x)
        
class ResConvUnits(nn.Module):
    def __init__(self,in_dim,nb_residual_unit,nb_filter=64):
        super(ResConvUnits,self).__init__()
        self.conv1 = nn.Conv2d(in_dim,64,3,1,1)
        self.resseq = ResSeq(64,nb_filter,nb_residual_unit)
        self.conv2 = nn.Conv2d(nb_filter,2,3,1,1)
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.resseq(out)
        out = F.relu(out)
        return self.conv2(out)
        
class External(nn.Module):
    def __init__(self,external_dim):
        super(External,self).__init__()
        self.embed = nn.Linear(external_dim,10)
        self.fc = nn.Linear(10,2)
    def forward(self,x):
        x = x.permute(0,2,3,1)
        #print("x",x.shape)
        out = self.embed(x)
        print(out.shape)
        out = F.relu(out)
        out = self.fc(out)
        print(out.shape)
        out = F.relu(out)
        return out
           
class ST_ResNet(nn.Module):
    def __init__(self,c_conf=(3,2,32,32),p_conf=(1,2,32,32),t_conf=(1,2,32,32),external_dim=8,nb_residual_unit=3):
        super(ST_ResNet,self).__init__()
        nb_flow,map_width,map_height = t_conf[1],t_conf[2],t_conf[3]
        self.closeness = ResConvUnits(c_conf[0]*c_conf[1],nb_residual_unit)
        self.period = ResConvUnits(p_conf[0]*p_conf[1],nb_residual_unit)
        self.trend = ResConvUnits(t_conf[0]*t_conf[1],nb_residual_unit)
        
        self.WC = nn.Parameter(torch.randn((1,nb_flow,c_conf[2],c_conf[3]),requires_grad=True))
        self.WP = nn.Parameter(torch.randn((1,nb_flow,p_conf[2],p_conf[3]),requires_grad=True))
        self.WT = nn.Parameter(torch.randn((1,nb_flow,t_conf[2],t_conf[3]),requires_grad=True))

        #self.external = External(external_dim)
        
        self.external = nn.Sequential(OrderedDict([
            ('embed', nn.Linear(external_dim, 10)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, nb_flow)),
            ('relu2', nn.ReLU())
            ]))
        #print(self.external)
    def forward(self,x):
        x_c,x_p,x_t,x_e = x[0],x[1],x[2],x[3]
        out1 = self.closeness(x_c)
        out2 = self.period(x_p)
        out3 = self.trend(x_t)
        out = torch.mul(out1,self.WC)+torch.mul(out2,self.WP)+torch.mul(out3,self.WT)
        ext = self.external(x_e).unsqueeze(2).unsqueeze(3)
        ext = torch.repeat_interleave(ext, repeats=32, dim=2)
        ext = torch.repeat_interleave(ext, repeats=32, dim=3)
    #ext.shape = ()
        #print("out:",out.shape,"ext:",ext.shape)
        # return 1
        # print(ext.shape,out.shape)
        ret = ext+out
        
        
        return F.tanh(ret)

# %%
# xc = torch.randn(100, 6,32, 32,requires_grad=True)
# xp = torch.randn(100, 2,32, 32,requires_grad=True)
# xt = torch.randn(100, 2,32, 32,requires_grad=True)
# xe = torch.randn(100, 8, requires_grad=True)
# y = torch.randn(100,2,32,32,requires_grad=True)
# x = [xc,xp,xt,xe]

# external_dim = 8
# c_conf =(3,2,32,32)
# p_conf = (1,2,32,32)
# t_conf = (1,2,32,32)
# nb_residual_unit = 3

# model = ST_ResNet()
# predict = model(x)


# %%
#predict.view(len(y),-1).shape
# %%
