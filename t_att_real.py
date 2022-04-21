#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import os
import sys


EPSILON = torch.finfo(torch.float32).eps



class T_att_real(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 16):
        super(T_att_real, self).__init__()
        self.query = nn.LSTM(in_channel, out_channel, dropout=0.1)
        self.key = nn.LSTM(in_channel, out_channel, dropout=0.1)
        self.value = nn.LSTM(in_channel, out_channel, dropout=0.1)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v):
        causal = True
        # NF * T * C
        query = self.query(q.transpose(0, 1))[0].transpose(0, 1)
        key = self.key(k.transpose(0, 1))[0].transpose(0, 1)
        value = self.value(v.transpose(0, 1))[0].transpose(0, 1)
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / 16**0.5
        if causal:
            mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2]), diagonal=0)
            mask = mask.to(energy.device)
            energy = energy * mask
        energy = self.softmax(energy) # NF * T * T
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        
        return weighted_value

class Self_Attention_T_real(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 16):
        super(Self_Attention_T_real, self).__init__()
        self.T_att = T_att_real(in_channel)

        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(out_channel)

    def forward(self, x):
        # N*F, T, C
        out = self.layernorm1(x)
        out = self.T_att(out, out, out)
        out = self.layernorm2(out)
        return out

class Multihead_Attention_T_Branch_real(nn.Module):
     def __init__(self, n_heads=1, in_channel = 64, out_channel = 16, spliceindex=[-4,-3,-2,-1,0,1,2,3,4]):
         super(Multihead_Attention_T_Branch_real, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_T_real(in_channel) for _ in range(n_heads)] )
         self.transformer_linear = nn.Linear(out_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T * 2
        
        N, C, F, T = inputs.shape
        x = inputs.permute(0, 2, 3, 1) # N F T C 2
        x = x.contiguous().view([N*F, T, C])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transformer_linear(x)
        outs = outs.contiguous().view([N, F, T, C]) 
        outs = self.prelu(self.layernorm3(outs))
        outs = self.dropout(outs)
        outs = outs.permute(0, 3, 1, 2)
        outs = outs + inputs
        return outs



if __name__ == '__main__':
    net = Multihead_Attention_T_Branch_real()
    inputs = torch.ones([10, 64, 32, 398])
    y = net(inputs)
    print(y.shape)
