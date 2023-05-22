# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from functools import reduce
from operator import mul
import random
import torch.nn.functional as F
import csv
import os
import sys
import time
import math


term_width = 100
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


# def set_parameters(meta_networks, grad_list, bias=False):
#     weight_offset0 = 0
#     bias_offset0 = 0
#     weight_offset1 = 0
#     bias_offset1 = 0
#     weight_offset2 = 0
#     bias_offset2 = 0
#     weight_offset3 = 0
#     bias_offset3 = 0
#     weight_offset4 = 0
#     bias_offset4 = 0
#     if bias == True:
#         for name, params in meta_networks.named_parameters():
#             if name == 'dw_weights0' or 'dw_weights1' or 'dw_weights2' or 'dw_weights3' or 'dw_weights4':
#                 weight_shape0 = params.size()
#                 weight_flat_size0 = reduce(mul, weight_shape0, 1)
#                 params.data = grad_list[weight_offset0:weight_offset0 + weight_flat_size0].view(*weight_shape0)
#                 weight_offset0 += weight_flat_size0
#             elif name == 'dw_weights0_bias' or 'dw_weights1_bias' or 'dw_weights2_bias' or 'dw_weights3v' or 'dw_weights4_bias':
#                 bias_shape0 = params.size()
#                 bias_flat_size0 = reduce(mul, bias_shape0, 1)
#                 params.data = grad_list[bias_offset0:bias_offset0 + bias_flat_size0].view(*bias_shape0)
#                 bias_offset0 += bias_flat_size0
#             elif name == 'up_sample0' or 'up_sample1' or 'up_sample2' or 'up_sample3':
#                 weight_shape1 = params.size()
#                 weight_flat_size1 = reduce(mul, weight_shape1, 1)
#                 params.data = grad_list[weight_offset1:weight_offset1 + weight_flat_size1].view(*weight_shape1)
#                 weight_offset1 += weight_flat_size1
#             elif name == 'up_sample0_bias' or 'up_sample1_bias' or 'up_sample2_bias' or 'up_sample3_bias':
#                 bias_shape1 = params.size()
#                 bias_flat_size1 = reduce(mul, bias_shape1, 1)
#                 params.data = grad_list[bias_offset1:bias_offset1 + bias_flat_size1].view(*bias_shape1)
#                 bias_offset1 += bias_flat_size1
#             elif name == 'up_weights0' or 'up_weights1' or 'up_weights2' or 'up_weights3':
#                 weight_shape2 = params.size()
#                 weight_flat_size2 = reduce(mul, weight_shape2, 1)
#                 params.data = grad_list[weight_offset2:weight_offset2 + weight_flat_size2].view(*weight_shape2)
#                 weight_offset2 += weight_flat_size2
#             elif name == 'up_weights0_bias' or 'up_weights1_bias' or 'up_weights2_bias' or 'up_weights3_bias':
#                 bias_shape2 = params.size()
#                 bias_flat_size2 = reduce(mul, bias_shape2, 1)
#                 params.data = grad_list[bias_offset2:bias_offset2 + bias_flat_size2].view(*bias_shape2)
#                 bias_offset2 += bias_flat_size2
#             elif name == 'gate_weights0':
#                 weight_shape3 = params.size()
#                 weight_flat_size3 = reduce(mul, weight_shape3, 1)
#                 params.data = grad_list[weight_offset3:weight_offset3 + weight_flat_size3].view(*weight_shape3)
#                 weight_offset3 += weight_flat_size3
#             elif name == 'gate_weights0_bias':
#                 bias_shape3 = params.size()
#                 bias_flat_size3 = reduce(mul, bias_shape3, 1)
#                 params.data = grad_list[bias_offset3:bias_offset3 + bias_flat_size3].view(*bias_shape3)
#                 bias_offset3 += bias_flat_size3
#             elif name == 'dw_bn0.weight' or 'dw_bn1.weight' or 'dw_bn2.weight' or 'dw_bn3.weight' or 'up_bn00.weight' or 'up_bn01.weight' or \
#                     'up_bn10.weight' or 'up_bn11.weight' or 'up_bn20.weight' or 'up_bn21.weight' or 'up_bn30.weight' or 'up_bn31.weight' or 'gate_bn0.weight':
#                 weight_shape4 = params.size()
#                 weight_flat_size4 = reduce(mul, weight_shape4, 1)
#                 params.data = grad_list[weight_offset4:weight_offset4 + weight_flat_size4].view(*weight_shape4)
#                 weight_offset4 += weight_flat_size4
#             else:
#                 bias_shape4 = params.size()
#                 bias_flat_size4 = reduce(mul, bias_shape4, 1)
#                 params.data = grad_list[bias_offset4:bias_offset4 + bias_flat_size4].view(*bias_shape4)
#                 bias_offset4 += bias_flat_size4
#     else:
#         for name, params in meta_networks.named_parameters():
#             if name == 'dw_weights0' or 'dw_weights1' or 'dw_weights2' or 'dw_weights3' or 'dw_weights4':
#                 weight_shape0 = params.size()  # params的shape，例如卷积层的[2, 4, 3, 3]
#                 weight_flat_size0 = reduce(mul, weight_shape0, 1)  # params的参数量，例如2*4*3*3=72
#                 params.data = grad_list[weight_offset0:weight_offset0 + weight_flat_size0].view(*weight_shape0)  # 对params中的参数进行更新
#                 weight_offset0 += weight_flat_size0
#             elif name == 'up_sample0' or 'up_sample1' or 'up_sample2' or 'up_sample3':
#                 weight_shape1 = params.size()
#                 weight_flat_size1 = reduce(mul, weight_shape1, 1)
#                 params.data = grad_list[weight_offset1:weight_offset1 + weight_flat_size1].view(*weight_shape1)
#                 weight_offset1 += weight_flat_size1
#             elif name == 'up_weights0' or 'up_weights1' or 'up_weights2' or 'up_weights3':
#                 weight_shape2 = params.size()
#                 weight_flat_size2 = reduce(mul, weight_shape2, 1)
#                 params.data = grad_list[weight_offset2:weight_offset2 + weight_flat_size2].view(*weight_shape2)
#                 weight_offset2 += weight_flat_size2
#             elif name == 'gate_weights0':
#                 weight_shape3 = params.size()
#                 weight_flat_size3 = reduce(mul, weight_shape3, 1)
#                 params.data = grad_list[weight_offset3:weight_offset3 + weight_flat_size3].view(*weight_shape3)
#                 weight_offset3 += weight_flat_size3
#             elif name == 'dw_bn0.weight' or 'dw_bn1.weight' or 'dw_bn2.weight' or 'dw_bn3.weight' or 'up_bn00.weight' or 'up_bn01.weight' or \
#                     'up_bn10.weight' or 'up_bn11.weight' or 'up_bn20.weight' or 'up_bn21.weight' or 'up_bn30.weight' or 'up_bn31.weight' or 'gate_bn0.weight':
#                 weight_shape4 = params.size()
#                 weight_flat_size4 = reduce(mul, weight_shape4, 1)
#                 params.data = grad_list[weight_offset4:weight_offset4 + weight_flat_size4].view(*weight_shape4)
#                 weight_offset4 += weight_flat_size4
#             else:
#                 bias_shape4 = params.size()
#                 bias_flat_size4 = reduce(mul, bias_shape4, 1)
#                 params.data = grad_list[bias_offset4:bias_offset4 + bias_flat_size4].view(*bias_shape4)
#                 bias_offset4 += bias_flat_size4


def set_parameters(meta_networks, grad_list):
    offset = 0
    for name, params in meta_networks.named_parameters():
        weight_shape = params.size()
        weight_flat_size = reduce(mul, weight_shape, 1)
        params.data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
        offset += weight_flat_size


def get_parameters(meta_learner):
    _loss_grad = []
    for name, params in meta_learner.named_parameters():
        _loss_grad.append(params.view(-1).unsqueeze(1))
    flat_loss_grad = torch.cat(_loss_grad, dim=0).view(1, -1)
    return flat_loss_grad


class CSV_writer():
    def __init__(self, path):
        self.file = open(path,'w',encoding='utf-8')
        self.writer=csv.writer(self.file)
    def Writer(self, content):
        self.writer.writerow([str(content.data.cpu().numpy())])
    def close(self):
        self.file.close()


def reset_parameters(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, a=-5, b=5)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-5, b=5)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
