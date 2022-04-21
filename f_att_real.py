#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import os
import sys


EPSILON = torch.finfo(torch.float32).eps



class F_att_real(nn.Module):
    def __init__(self, in_channel=64, out_channel=16):
        super(F_att_real, self).__init__()
        self.query = nn.LSTM(in_channel, out_channel, dropout=0.1, bidirectional=True)
        self.key = nn.LSTM(in_channel, out_channel, dropout=0.1, bidirectional=True)
        self.value = nn.LSTM(in_channel, out_channel, dropout=0.1, bidirectional=True)
        self.querylinear = nn.Linear(out_channel*2, out_channel)
        self.keylinear = nn.Linear(out_channel*2, out_channel)
        self.valuelinear = nn.Linear(out_channel*2, out_channel)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v):
        # NT * F * C
        query = self.query(q.transpose(0, 1))[0].transpose(0, 1)
        key = self.key(k.transpose(0, 1))[0].transpose(0, 1)
        value = self.value(v.transpose(0, 1))[0].transpose(0, 1)
        query = self.querylinear(query)
        key = self.keylinear(key)
        value = self.valuelinear(value)
        # output = []
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / 16**0.5
        energy = self.softmax(energy) # NT * F * F
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])

        return weighted_value

class Self_Attention_F_real(nn.Module):
    def __init__(self, in_channel=64, out_channel=16):
        super(Self_Attention_F_real, self).__init__()
        self.F_att = F_att_real(in_channel=in_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(out_channel)

    def forward(self, x):
        # N*T, F, C, 2
        out = self.layernorm1(x)
        out = self.F_att(out, out, out)
        out = self.layernorm2(out)
        return out

class Multihead_Attention_F_Branch_real(nn.Module):
     def __init__(self, n_heads=1, in_channel=64, out_channel=16, spliceindex=[-4,-3,-2,-1,0,1,2,3,4]):
         super(Multihead_Attention_F_Branch_real, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_F_real(in_channel=in_channel) for _ in range(n_heads)] )
         self.transformer_linear = nn.Linear(out_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T 
        
        N, C, F, T = inputs.shape
        x = inputs.permute(0, 3, 2, 1) # N T F C 2
        x = x.contiguous().view([N*T, F, C])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)

        out = self.transformer_linear(x)
        
        out = out.contiguous().view([N, T, F, C]) 
        out = out.permute(0, 3, 2, 1)
        out = self.prelu(self.layernorm3(out.transpose(1, 3)).transpose(1, 3))
        out = self.dropout(out)
        out = out + inputs
        return out

if __name__ == '__main__':
    net = Multihead_Attention_F_Branch_real()
    inputs = torch.ones([10, 64, 32, 398])
    y = net(inputs)
    print(y.shape)
