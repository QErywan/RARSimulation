# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:51:34 2020

@author: HOME-PC
"""

# import torch

# def get_device(gpu):
#     if gpu == 0: # use all available GPUs (was if gpu = -1)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         device_ids = [x for x in range(torch.cuda.device_count())]
#     else:
#         if gpu > torch.cuda.device_count()-1:
#             # print(gpu, torch.cuda.device_count())
#             raise Exception('got gpu index={}, but there are only {} GPUs'.format(gpu, torch.cuda.device_count()))
#         if torch.cuda.is_available():
#             device = 'cuda:{}'.format(gpu)
#             device_ids = [gpu]
#         else:
#             device = 'cpu'

#     if device == 'cpu':
#         print('*** Warning: No GPU was found, running over CPU.')

#     print('*** Set device to {}'.format(device))
#     if device == 'cuda' and torch.cuda.device_count() > 1:
#         print('*** Running on multiple GPUs ({})'.format(torch.cuda.device_count()))

#     return device, device_ids

import torch

def get_device(gpu):
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # macOS with MPS (Metal Performance Shaders)
        device = 'mps'
        device_ids = [0]  # Only 1 MPS device is available at the moment
    elif gpu == 0:  # use all available GPUs
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_ids = [x for x in range(torch.cuda.device_count())]
    else:
        if gpu > torch.cuda.device_count() - 1:
            raise Exception('got gpu index={}, but there are only {} GPUs'.format(gpu, torch.cuda.device_count()))
        if torch.cuda.is_available():
            device = 'cuda:{}'.format(gpu)
            device_ids = [gpu]
        else:
            device = 'cpu'

    if device == 'cpu':
        print('*** Warning: No GPU or MPS device was found, running on CPU.')

    print('*** Set device to {}'.format(device))
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print('*** Running on multiple GPUs ({})'.format(torch.cuda.device_count()))

    return device, device_ids
