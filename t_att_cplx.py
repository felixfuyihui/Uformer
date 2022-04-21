#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import os
import sys


EPSILON = torch.finfo(torch.float32).eps



class T_att(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 16):
        super(T_att, self).__init__()
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

class Self_Attention_T(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 16):
        super(Self_Attention_T, self).__init__()
        self.T_att1 = T_att(in_channel=in_channel)
        self.T_att2 = T_att(in_channel=in_channel)
        self.T_att3 = T_att(in_channel=in_channel)
        self.T_att4 = T_att(in_channel=in_channel)
        self.T_att5 = T_att(in_channel=in_channel)
        self.T_att6 = T_att(in_channel=in_channel)
        self.T_att7 = T_att(in_channel=in_channel)
        self.T_att8 = T_att(in_channel=in_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(out_channel)

    def forward(self, x):
        # N*F, T, C, 2
        x = self.layernorm1(x.transpose(2, 3)).transpose(2, 3)
        real, imag = x[...,0], x[...,1]
        A = self.T_att1(real, real, real)
        B = self.T_att2(real, imag, imag)
        C = self.T_att3(imag, real, imag)
        D = self.T_att4(imag, imag, real)
        E = self.T_att5(real, real, imag)
        F = self.T_att6(real, imag, real)
        G = self.T_att7(imag, real, real)
        H = self.T_att8(imag, imag, imag)
        real_att = A-B-C-D
        imag_att = E+F+G-H
        out = torch.stack([real_att, imag_att], -1)
        out = self.layernorm2(out.transpose(2, 3)).transpose(2, 3)
        return out

class Multihead_Attention_T_Branch(nn.Module):
     def __init__(self, n_heads=1, in_channel = 64, out_channel = 16, spliceindex=[-4,-3,-2,-1,0,1,2,3,4]):
         super(Multihead_Attention_T_Branch, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_T(in_channel=in_channel) for _ in range(n_heads)] )
         self.transformer_linear_real = nn.Linear(out_channel, in_channel)
         self.transformer_linear_imag = nn.Linear(out_channel, in_channel)
         self.layernorm3 = nn.LayerNorm(in_channel)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T * 2
        
        N, C, F, T, ri = inputs.shape
        x = inputs.permute(0, 2, 3, 1, 4) # N F T C 2
        x = x.contiguous().view([N*F, T, C, ri])
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        x_real, x_imag = x[...,0], x[...,1]

        out_real = self.transformer_linear_real(x_real) - self.transformer_linear_imag(x_imag)
        out_imag = self.transformer_linear_real(x_imag) + self.transformer_linear_imag(x_real)
        out_real = out_real.contiguous().view([N, F, T, C]) 
        out_imag = out_imag.contiguous().view([N, F, T, C])
        out_real = out_real.permute(0, 3, 1, 2)
        out_imag = out_imag.permute(0, 3, 1, 2)
        outs = torch.stack([out_real, out_imag], -1)
        outs = self.prelu(self.layernorm3(outs.transpose(1, 4)).transpose(1, 4))
        outs = self.dropout(outs)
        outs = outs + inputs
        return outs


class Self_Attention_T_RICAT(nn.Module):
    def __init__(self, in_channel = 128, out_channel = 32):
        super(Self_Attention_T_RICAT, self).__init__()
        self.T_att = T_att(in_channel, out_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(out_channel)

    def forward(self, x):
        # N*F, T, 2C
        out = self.layernorm1(x)
        out = self.T_att(out, out, out)
        out = self.layernorm2(out)
        return out


class Multihead_Attention_T_Branch_RICAT(nn.Module):
     def __init__(self, n_heads=1, in_channel = 128, out_channel = 32, spliceindex=[-4,-3,-2,-1,0,1,2,3,4]):
         super(Multihead_Attention_T_Branch_RICAT, self).__init__()
         self.attn_heads = nn.ModuleList([Self_Attention_T_RICAT() for _ in range(n_heads)] )
         self.transformer_linear_real = nn.Linear(out_channel//2, in_channel//2)
         self.transformer_linear_imag = nn.Linear(out_channel//2, in_channel//2)
         self.layernorm3 = nn.LayerNorm(in_channel//2)
         self.dropout = nn.Dropout(p=0.1)
         self.prelu = nn.PReLU()

     def forward(self, inputs):
        # N * C * F * T * 2
        N, C, F, T, ri = inputs.shape
        x = inputs.permute(0, 2, 3, 1, 4) # N F T C 2
        x = x.contiguous().view([N*F, T, C, ri])
        x = torch.cat([x[...,0], x[...,1]], -1)
        x = [attn(x) for i, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        x_real, x_imag = x.chunk(2, -1)
        out_real = self.transformer_linear_real(x_real) - self.transformer_linear_imag(x_imag)
        out_imag = self.transformer_linear_real(x_imag) + self.transformer_linear_imag(x_real)
        out_real = out_real.contiguous().view([N, F, T, C]) 
        out_imag = out_imag.contiguous().view([N, F, T, C])
        out_real = out_real.permute(0, 3, 1, 2)
        out_imag = out_imag.permute(0, 3, 1, 2)
        outs = torch.stack([out_real, out_imag], -1)
        outs = self.prelu(self.layernorm3(outs.transpose(1, 4)).transpose(1, 4))
        outs = self.dropout(outs)
        outs = outs + inputs
        return outs


if __name__ == '__main__':
    net = Multihead_Attention_T_Branch_RICAT()
    inputs = torch.ones([10, 64, 32, 398, 2])
    y = net(inputs)
    print(y.shape)
