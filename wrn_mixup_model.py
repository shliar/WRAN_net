### dropout has been removed in this code. original code had dropout#####
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
act = torch.nn.ReLU()


import math
from torch.nn.utils.weight_norm import WeightNorm


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.inplanes = 64
        #print('out_planes',out_planes)
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
        #                        padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=5, stride=1,
        #                        padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        # print('x',self.convShortcut(x).shape)
        # print('out',out.shape)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7,num_attention_heads=1, input_size=84, hidden_size=84):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.input_size11=input_size
        if self.input_size11 >0:
            self.key_layer = nn.Linear(input_size, hidden_size)
            self.query_layer = nn.Linear(input_size, hidden_size)
            self.value_layer = nn.Linear(input_size, hidden_size)
    
    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        print(x.shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # print('111',x.shape)
        if self.input_size11 >0:
            key = self.key_layer(x)
            query = self.query_layer(x)
            value = self.value_layer(x)
            key_heads = key
            query_heads = query
            value_heads = value

            attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_probs = F.softmax(attention_scores, dim = -1)

            context = torch.matmul(attention_probs, value_heads)
            context = context.permute(0, 2, 1, 3).contiguous()
            new_size = context.size()[ : -2] + (self.all_head_size , )
            context = context.view(*new_size)[:, None, :,:]
            # print('222',self.sigmoid(context).shape)
            # print('333',self.sigmoid(x).shape)
            return self.sigmoid(context)
        else:
            return self.sigmoid(x)



class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores   
    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


def to_one_hot(inp,num_classes):

    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    if torch.cuda.is_available():
        y_onehot = y_onehot.cuda()

    y_onehot.zero_()
    x = inp.type(torch.LongTensor)
    if torch.cuda.is_available():
        x = x.cuda()

    x = torch.unsqueeze(x , 1)
    y_onehot.scatter_(1, x , 1)
    
    return Variable(y_onehot,requires_grad=False)
    # return y_onehot


def mixup_data(x, y, lam):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
   
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

    
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes= 200 , loss_type = 'dist', per_img_std = False, stride = 1 ):
        dropRate = 0.5
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        self.inplanes=64
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=7, stride=1,
                               padding=3, bias=False)
        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(nChannels[0])
        self.sa = SpatialAttention(kernel_size=7)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        #eca block
        #self.eca_block1 =eca_block(nChannels[1])
        #self.eca_block2 =eca_block(nChannels[2])
        #self.eca_block3 =eca_block(nChannels[3])
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(nChannels[3])
        self.sa1 = SpatialAttention(kernel_size=7,input_size=1, hidden_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        
        if loss_type == 'softmax':
            self.linear = nn.Linear(nChannels[3], int(num_classes))
            self.linear.bias.data.fill_(0)
        else:
            self.linear = distLinear(nChannels[3], int(num_classes))
        
        self.num_classes = num_classes
        if flatten:
            self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x, target= None, mixup=False, mixup_hidden=True, mixup_alpha=None , lam = 0.4):
        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   

            out = x

            target_a = target_b  = target

            if layer_mix == 0:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

            out = self.conv1(out)
            out = self.ca(out) * out
            # print('ca:{}'.format(out.shape))
            out = self.sa(out) * out
            # print('sa:{}'.format(out.shape))
            # exit()
            out = self.maxpool(out)
            #out = self.eca_block1(out)
            
            out = self.block1(out)


            if layer_mix == 1:
                out, target_a , target_b , lam  = mixup_data(out, target, lam=lam)
            #out = self.eca_block2(out)
            out = self.block2(out)
            
            if layer_mix == 2:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

            #out = self.eca_block3(out)
            out = self.block3(out)
            
            if  layer_mix == 3:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = self.ca1(out) * out
            # print('ca:{}'.format(out.shape))
            # print('sa:{}'.format(self.sa1(out).shape))
            # exit()
            out = self.sa1(out) * out
            
            
            out = self.avgpool(out)
            out = out.reshape(out.size(0), -1)
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)
            
            
            return out , out1 , target_a , target_b
        else: 
            out = x
            out = self.conv1(out)
            out = self.ca(out) * out
            out = self.sa(out) * out
            #out = self.attention(out)
            out = self.maxpool(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = self.ca1(out) * out
            out = self.sa1(out) * out
            #out = self.attention1(out)
            out = self.avgpool(out)
            out = out.reshape(out.size(0), -1)
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)
            
            
            return out, out1
        
                  
        
def wrn28_10(num_classes=200 , loss_type = 'dist'):
    model = WideResNet(depth=28, widen_factor=10, num_classes=12, loss_type = loss_type , per_img_std = False, stride = 1 )
    return model

