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
from time import time

EPSILON = torch.finfo(torch.float32).eps
sys.path.append(os.path.dirname(sys.path[0]) + '/model')

from trans import STFT, iSTFT, MelTransform, inv_MelTransform
from conv2d_cplx import ComplexConv2d_Encoder, ComplexConv2d_Decoder
from conv2d_real import RealConv2d_Encoder, RealConv2d_Decoder
from dilated_dualpath_conformer import Dilated_Dualpath_Conformer
from fusion import fusion as fusion
from show import show_model, show_params


def tanhextern(input):
    out = 10 * (1-torch.exp(-0.1*input)) / (1+torch.exp(-0.1*input))

class Uformer(nn.Module):

    def __init__(self, 
                win_len=400, 
                win_inc=160, 
                fft_len=512, 
                win_type='hanning', 
                fid=None):
        super(Uformer, self).__init__()
        input_dim = win_len
        output_dim = win_len
        self.kernel_num = [1,8,16,32,64,128,128]
        self.kernel_num_real = [1,8,16,32,64,128]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder_real = nn.ModuleList()
        self.decoder_real = nn.ModuleList()
        for idx in range(len(self.kernel_num)-1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d_Encoder(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                        dilation=(1, 1),
                        groups = 1
                    ),
                    nn.BatchNorm3d(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )

        for idx in range(len(self.kernel_num)-1):
            self.encoder_real.append(
                nn.Sequential(
                    RealConv2d_Encoder(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2,1),
                        dilation=(1, 1),
                        groups = 1
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )

        
        self.conformer = Dilated_Dualpath_Conformer()


        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx >= 2:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConv2d_Decoder(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(5, 2),
                        stride=(2,1),
                        padding=(2, 0),
                        output_padding = (1, 0),
                        dilation=(1, 1),
                        groups = 1
                    ),  
                    nn.BatchNorm3d(self.kernel_num[idx-1]),
                    #nn.ELU()
                    nn.PReLU()
                    )
                )
          
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConv2d_Decoder(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(5, 2),
                        stride=(2,1),
                        padding=(2, 0),
                        output_padding = (1, 0),
                        dilation=(1, 1),
                        groups = 1
                    ),  
                    )
                )

        for idx in range(len(self.kernel_num)-1, 0, -1):
            if idx >= 2:
                self.decoder_real.append(
                    nn.Sequential(
                        RealConv2d_Decoder(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(5, 2),
                        stride=(2,1),
                        padding=(2,0),
                        output_padding = (1, 0),
                        dilation=(1, 1),
                        groups = 1
                    ),  
                    nn.BatchNorm2d(self.kernel_num[idx-1]),
                    #nn.ELU()
                    nn.PReLU()
                    )
                )
        
            else:
                self.decoder_real.append(
                    nn.Sequential(
                        RealConv2d_Decoder(
                        self.kernel_num[idx]*2,
                        self.kernel_num[idx-1],
                        kernel_size =(5, 2),
                        stride=(2,1),
                        padding=(2,0),
                        output_padding = (1, 0),
                        dilation=(1, 1),
                        groups = 1
                    ),  
                    )
                )

                

        self.stft = STFT(frame_len=win_len, frame_hop=win_inc)
        self.istft = iSTFT(frame_len=win_len, frame_hop=win_inc)

        show_model(self, fid)
        show_params(self, fid)

    def flatten_parameters(self):
        self.enhance.flatten_parameters()

    def forward(self, inputs, src):
        warnings.filterwarnings('ignore')
        

        inputs_real, inputs_imag = self.stft(inputs[:,0].unsqueeze(1))
        src_real, src_imag = self.stft(src[:,0])
        src = self.istft((src_real, src_imag))
        src_mag, src_pha = torch.sqrt(torch.clamp(src_real ** 2 + src_imag ** 2, EPSILON)), torch.atan2(src_imag+EPSILON, src_real)

        src_mag = src_mag ** 0.5
        src_real, src_imag = src_mag * torch.cos(src_pha), src_mag * torch.sin(src_pha)
        src_cplx = torch.stack([src_real, src_imag], 1)
        

        mag, phase = torch.sqrt(torch.clamp(inputs_real ** 2 + inputs_imag ** 2, EPSILON)), torch.atan2(inputs_imag+EPSILON, inputs_real)
        mag = mag ** 0.5
        mag_input = []

        mag_input.append(mag)

        

        
        inputs_real, inputs_imag = mag * torch.cos(phase), mag * torch.sin(phase)



    

        out = torch.stack([inputs_real, inputs_imag], -1) # B C F T 2
        out = out[:, :, 1:]
        mag = mag[:, :, 1:]
        encoder_out = []
        mag_out = []

        for idx in range(len(self.encoder)):
            out = self.encoder[idx](out)
            mag = self.encoder_real[idx](mag)
            out, mag = fusion(out, mag)
            mag_out.append(mag)
            encoder_out.append(out)



        out, mag = self.conformer(out, mag)
        
        for idx in range(len(self.decoder)):
            out_cat = torch.cat([encoder_out[-1 - idx],out],1)
            out = self.decoder[idx](out_cat)
         
            mag_cat = torch.cat([mag_out[-1 - idx],mag],1)
            mag = self.decoder_real[idx](mag_cat)

            out, mag = fusion(out, mag)



        mag = torch.sigmoid(mag)
        mag = F.pad(mag, [0,0,1,0])

        mag = mag[:,0] * mag_input[0][:,0]

        mask_real = out[...,0]
        mask_imag = out[...,1]
    
        mask_mags = torch.sqrt(torch.clamp(mask_real**2 + mask_imag**2, EPSILON))
        real_phase = mask_real/(mask_mags+EPSILON)
        imag_phase = mask_imag/(mask_mags+EPSILON)
        mask_mags = torch.tanh(mask_mags+EPSILON)
        mask_phase = torch.atan2(imag_phase+EPSILON, real_phase)
        mask_mags = F.pad(mask_mags, [0,0,1,0])
        mask_phase = F.pad(mask_phase, [0,0,1,0])


        
        est_mags = mask_mags[:, 0]*mag_input[0][:,0]


        est_phase = phase[:, 0] + mask_phase[:, 0]
        


        mag_compress, pha_compress = est_mags, est_phase
        mag_compress = (mag_compress + mag)*0.5

        real, imag = mag_compress * torch.cos(pha_compress), mag_compress * torch.sin(pha_compress)
        
        output_real = []
        output_imag = []
        output = []
        output_real.append(real)
        output_imag.append(imag)


        mag_compress = mag_compress ** 2
        real, imag = mag_compress * torch.cos(pha_compress), mag_compress * torch.sin(pha_compress)


        spk1 = self.istft((real, imag))
        output.append(spk1)

        output = torch.stack(output, 1)
        output = output.squeeze(1)
        output_real = torch.stack(output_real, 1)
        output_imag = torch.stack(output_imag, 1)
        output_real = output_real.squeeze(1) # N x C x F x T
        output_imag = output_imag.squeeze(1)
        output_cplx = torch.stack([output_real, output_imag], 1) # N x 2 x C x F x T
        return output, src, output_cplx, src_cplx

    def get_params(self, weight_decay=0.0):

        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params


if __name__ == '__main__':
    torch.manual_seed(10)
    torch.set_num_threads(4)

    import soundfile as sf 
    import numpy as np

    net = Uformer()
    inputs = torch.randn([10,1,64000])
    print(inputs.shape)
    outputs = net(inputs,inputs)


