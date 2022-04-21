#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

import torch_complex
from torch_complex import ComplexTensor
import warnings

EPSILON = torch.finfo(torch.float32).eps
sys.path.append(os.path.dirname(sys.path[0]) + '/model')

from f_att_cplx import Multihead_Attention_F_Branch
from t_att_cplx import Multihead_Attention_T_Branch
from f_att_real import Multihead_Attention_F_Branch_real
from t_att_real import Multihead_Attention_T_Branch_real
from dsconv2d_cplx import DSConv2d
from dsconv2d_real import DSConv2d_Real
from fusion import fusion as fusion
from show import show_model, show_params


class Dilated_Dualpath_Conformer(nn.Module):

    def __init__(self, channel=64):
        super(Dilated_Dualpath_Conformer, self).__init__()
        
        lstm_dim = 128
        lstm_layer = 3
        
        self.cplx_tatt = Multihead_Attention_T_Branch(in_channel=channel)
        self.cplx_fatt = Multihead_Attention_F_Branch(in_channel=channel)
        self.mag_tatt = Multihead_Attention_T_Branch_real(in_channel=channel)
        self.mag_fatt = Multihead_Attention_F_Branch_real(in_channel=channel)
        # self.cplx_dsconv = DSConv2d(dilation = dilation)
        # self.mag_dsconv = DSConv2d_Real(dilation = dilation)
        self.ln_conformer_cplx = nn.LayerNorm(channel)
        self.ln_conformer_mag = nn.LayerNorm(channel)


        # self.enhance_real_t = nn.LSTM(
        #             input_size= channel,
        #             hidden_size=128,
        #             num_layers=3,
        #             dropout=0.1,
        #     )
        # self.enhance_imag_t = nn.LSTM(
        #             input_size= channel,
        #             hidden_size=128,
        #             num_layers=3,
        #             dropout=0.1,
        #     )
        dilation = [1, 3, 5, 9]
        self.dsconv_cplx = nn.ModuleList()
        for idx in range(4):
            self.dsconv_cplx.append(DSConv2d(channel, channel//2, dilation=dilation[idx]))


        # self.tranform_real_t = nn.Linear(128, channel)
        # self.tranform_imag_t = nn.Linear(128, channel)

        self.enhance_real_f = nn.LSTM(
                    input_size= channel,
                    hidden_size=lstm_dim,
                    num_layers=lstm_layer,
                    dropout=0.1,
                    bidirectional=True
            )
        self.enhance_imag_f = nn.LSTM(
                    input_size= channel,
                    hidden_size=lstm_dim,
                    num_layers=lstm_layer,
                    dropout=0.1,
                    bidirectional=True
            )

        self.tranform_real_f = nn.Linear(lstm_dim*2, channel)
        self.tranform_imag_f = nn.Linear(lstm_dim*2, channel)


        # self.enhance_mag_t = nn.LSTM(
        #             input_size= channel,
        #             hidden_size=128,
        #             num_layers=3,
        #             dropout=0.1,
        #     )
        # self.tranform_mag_t = nn.Linear(128, channel)
        self.dsconv_real = nn.ModuleList()
        for idx in range(4):
            self.dsconv_real.append(DSConv2d_Real(channel, channel//2, dilation=dilation[idx]))
        self.enhance_mag_f = nn.LSTM(
                    input_size= channel,
                    hidden_size=lstm_dim,
                    num_layers=lstm_layer,
                    dropout=0.1,
                    bidirectional=True
            )
        self.tranform_mag_f = nn.Linear(lstm_dim*2, channel)
        self.prelu = nn.PReLU()


    def forward(self, cplx, mag):
        # N C F T 2
        # N C F T

        batch_size, channels, dims, lengths, ri = cplx.size()

        cplx_origin, mag_origin = [cplx], [mag]
        cplx = self.cplx_tatt(cplx)
        mag= self.mag_tatt(mag)
        cplx, mag = fusion(cplx, mag)

        for idx in range(len(self.dsconv_cplx)):
            cplx = self.dsconv_cplx[idx](cplx)
            mag = self.dsconv_real[idx](mag)
        cplx, mag = fusion(cplx, mag)
        cplx, mag = cplx + cplx_origin[0], mag + mag_origin[0]

        cplx_origin, mag_origin = [cplx], [mag]
        cplx = self.cplx_fatt(cplx)
        mag = self.mag_fatt(mag)
        cplx, mag = fusion(cplx, mag)

        cplx, mag = cplx.permute(2, 0, 3, 1, 4), mag.permute(2, 0, 3, 1)
        cplx = torch.reshape(cplx, [dims, batch_size*lengths, channels, ri])
        cplx_r = self.enhance_real_f(cplx[...,0])[0] - self.enhance_imag_f(cplx[...,1])[0]
        cplx_i = self.enhance_real_f(cplx[...,1])[0] + self.enhance_imag_f(cplx[...,0])[0]
        cplx = torch.stack([cplx_r, cplx_i], -1)
        cplx = self.prelu(cplx)
        cplx_r = self.tranform_real_f(cplx[...,0]) - self.tranform_imag_f(cplx[...,1])
        cplx_i = self.tranform_real_f(cplx[...,1]) + self.tranform_imag_f(cplx[...,0])
        cplx = torch.stack([cplx_r, cplx_i], -1)
        cplx = self.prelu(cplx)
        cplx = torch.reshape(cplx, [dims, batch_size, lengths, channels, ri])
        cplx = cplx.permute(1, 3, 0, 2, 4) # B C F T
        mag = torch.reshape(mag, [dims, batch_size*lengths, channels])
        mag = self.enhance_mag_f(mag)[0]
        mag = self.prelu(mag)
        mag = self.tranform_mag_f(mag)
        mag = self.prelu(mag)
        mag = torch.reshape(mag, [dims, batch_size, lengths, channels])
        mag = mag.permute(1, 3, 0, 2) # B C F T
        cplx, mag = fusion(cplx, mag)
        cplx, mag = cplx + cplx_origin[0], mag + mag_origin[0]

        # cplx = self.cplx_dsconv(cplx)
        # mag = self.mag_dsconv(mag)
        # cplx, mag = fusion(cplx, mag)
        cplx, mag = self.ln_conformer_cplx(cplx.transpose(1,4)).transpose(1,4), self.ln_conformer_mag(mag.transpose(1,3)).transpose(1,3)
        return cplx, mag

# class Conformer_Fusion(nn.Module):

#     def __init__(self, dilation = 4):
#         super(Conformer_Fusion, self).__init__()
       
#         self.mag_tatt = Multihead_Attention_T_Branch_real()
#         self.mag_fatt = Multihead_Attention_F_Branch_real()
#         self.ln_conformer_mag = nn.LayerNorm(64)

#         self.enhance_mag_t = nn.LSTM(
#                     input_size= 64,
#                     hidden_size=128,
#                     num_layers=3,
#                     dropout=0.1,
#             )
#         self.tranform_mag_t = nn.Linear(128, 64)
#         self.enhance_mag_f = nn.LSTM(
#                     input_size= 64,
#                     hidden_size=128,
#                     num_layers=3,
#                     dropout=0.1,
#                     bidirectional=True
#             )
#         self.tranform_mag_f = nn.Linear(128*2, 64)
#         self.prelu = nn.PReLU()


#     def forward(self, mag):
#         # N C F T 2
#         # N C F T

#         batch_size, channels, dims, lengths = mag.size()

#         mag= self.mag_tatt(mag)
#         mag = self.mag_fatt(mag)


#         mag = mag.permute(3, 0, 2, 1) # T B F C
#         mag = torch.reshape(mag, [lengths, batch_size*dims, channels])
#         mag = self.enhance_mag_t(mag)[0]
#         mag = self.prelu(mag)
#         mag = self.tranform_mag_t(mag)
#         mag = self.prelu(mag)
#         mag = torch.reshape(mag, [lengths, batch_size, dims, channels])
#         mag = mag.permute(2, 1, 0, 3) # F B T C
    
#         mag = torch.reshape(mag, [dims, batch_size*lengths, channels])
#         mag = self.enhance_mag_f(mag)[0]
#         mag = self.prelu(mag)
#         mag = self.tranform_mag_f(mag)
#         mag = self.prelu(mag)
#         mag = torch.reshape(mag, [dims, batch_size, lengths, channels])
#         mag = mag.permute(1, 3, 0, 2) # B C F T
#         mag = self.ln_conformer_mag(mag.transpose(1,3)).transpose(1,3)
#         return mag
if __name__ == '__main__':
    torch.manual_seed(10)
    inputs = torch.ones([10, 16, 12, 398, 2])
    net = Conformer_Fusion(16)
    outputs = net(inputs,inputs[...,0])
    print(outputs[0].shape, outputs[1].shape)
