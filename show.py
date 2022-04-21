#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def show_params(nnet, fid):
    
    print("=" * 40, "Model Parameters", "=" * 40)
    if fid is not None:
        fid.write("=" * 40+ "Model Parameters"+ "=" * 40 +"\n")
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                if fid is not None:
                    fid.write(str(name)+ str(params.size())+'\n')
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)
    if fid is not None:
        fid.write('[*] Parameter Size: {}'.format(num_params)+'\n')
        fid.write("=" * 98+'\n')
        fid.flush()


def show_model(nnet, fid):
    print("=" * 40, "Model Structures", "=" * 40)
    if fid is not None:
        fid.write("=" * 40+ "Model Structures"+"=" * 40+'\n')
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)
            if fid is not None:
                fid.write(str(m))
    print("=" * 98)
    if fid is not None:
        fid.write("=" * 98+'\n')
        fid.flush()
