import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layer import Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, Convolution6, Convolution7, Convolution8, Convolution9, \
    Convolution10, Convolution11, Convolution12, Convolution13, Convolution14, Convolution15, Convolution16, Convolution17, Fully_Connection


class ResNet18(nn.Module):
    def __init__(self, in_channel=3, c_in0=64, c_in1=128, c_in2=256, c_in3=512, f_in=512, num_classes=10):
        super(ResNet18, self).__init__()
        self.c_in0 = c_in0
        self.c_in1 = c_in1
        self.c_in2 = c_in2
        self.c_in3 = c_in3
        self.f_in = f_in
        self.c_layer1 = Convolution1(in_channel, c_in0)
        self.c_layer2 = Convolution2(c_in0, c_in0)
        self.c_layer3 = Convolution3(c_in0, c_in0)
        self.c_layer4 = Convolution4(c_in0, c_in0)
        self.c_layer5 = Convolution5(c_in0, c_in0)
        self.c_layer6 = Convolution6(c_in0, c_in1)
        self.c_layer7 = Convolution7(c_in1, c_in1)
        self.c_layer8 = Convolution8(c_in1, c_in1)
        self.c_layer9 = Convolution9(c_in1, c_in1)
        self.c_layer10 = Convolution10(c_in1, c_in2)
        self.c_layer11 = Convolution11(c_in2, c_in2)
        self.c_layer12 = Convolution12(c_in2, c_in2)
        self.c_layer13 = Convolution13(c_in2, c_in2)
        self.c_layer14 = Convolution14(c_in2, c_in3)
        self.c_layer15 = Convolution15(c_in3, c_in3)
        self.c_layer16 = Convolution16(c_in3, c_in3)
        self.c_layer17 = Convolution17(c_in3, c_in3)
        self.f_layer18 = Fully_Connection(f_in, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input, pr, pattern):
        if pattern == 0:
            weight_mean1 = self.c_layer1(input, 0, 0)
            weight_mean2 = self.c_layer2(input, 0, 0)
            weight_mean3 = self.c_layer3(input, 0, 0)
            weight_mean4 = self.c_layer4(input, 0, 0)
            weight_mean5 = self.c_layer5(input, 0, 0)
            weight_mean6, shortcut_weight_mean6 = self.c_layer6(input, 0, 0, 0)
            weight_mean7 = self.c_layer7(input, 0, 0)
            weight_mean8 = self.c_layer8(input, 0, 0)
            weight_mean9 = self.c_layer9(input, 0, 0)
            weight_mean10, shortcut_weight_mean10 = self.c_layer10(input, 0, 0, 0)
            weight_mean11 = self.c_layer11(input, 0, 0)
            weight_mean12 = self.c_layer12(input, 0, 0)
            weight_mean13 = self.c_layer13(input, 0, 0)
            weight_mean14, shortcut_weight_mean14 = self.c_layer14(input, 0, 0, 0)
            weight_mean15 = self.c_layer15(input, 0, 0)
            weight_mean16 = self.c_layer16(input, 0, 0)
            weight_mean17 = self.c_layer17(input, 0, 0)
            weight_mean18 = self.f_layer18(input, 0, 0)
            weight_mean = torch.cat((weight_mean1, weight_mean2, weight_mean3, weight_mean4, weight_mean5, weight_mean6, shortcut_weight_mean6, weight_mean7,
                                     weight_mean8, weight_mean9, weight_mean10, shortcut_weight_mean10, weight_mean11, weight_mean12, weight_mean13, weight_mean14,
                                     shortcut_weight_mean14, weight_mean15, weight_mean16, weight_mean17, weight_mean18), dim=0).view(1, 1, -1, 64)  # [1, 1, 83, 64]
            return weight_mean
        elif pattern == 1:
            c_pr1 = pr[0:self.c_in0]
            c_pr2 = pr[self.c_in0:2*self.c_in0]
            c_pr3 = pr[2*self.c_in0:3*self.c_in0]
            c_pr4 = pr[3*self.c_in0:4*self.c_in0]
            c_pr5 = pr[4*self.c_in0:5*self.c_in0]
            c_pr6 = pr[5*self.c_in0:5*self.c_in0+self.c_in1]
            c_shortcut_pr6 = pr[5*self.c_in0+self.c_in1:5*self.c_in0+2*self.c_in1]
            c_pr7 = pr[5*self.c_in0+2*self.c_in1:5*self.c_in0+3*self.c_in1]
            c_pr8 = pr[5*self.c_in0+3*self.c_in1:5*self.c_in0+4*self.c_in1]
            c_pr9 = pr[5*self.c_in0+4*self.c_in1:5*self.c_in0+5*self.c_in1]
            c_pr10 = pr[5*self.c_in0+5*self.c_in1:5*self.c_in0+5*self.c_in1+self.c_in2]
            c_shortcut_pr10 = pr[5*self.c_in0+5*self.c_in1+self.c_in2:5*self.c_in0+5*self.c_in1+2*self.c_in2]
            c_pr11 = pr[5*self.c_in0+5*self.c_in1+2*self.c_in2:5*self.c_in0+5*self.c_in1+3*self.c_in2]
            c_pr12 = pr[5*self.c_in0+5*self.c_in1+3*self.c_in2:5*self.c_in0+5*self.c_in1+4*self.c_in2]
            c_pr13 = pr[5*self.c_in0+5*self.c_in1+4*self.c_in2:5*self.c_in0+5*self.c_in1+5*self.c_in2]
            c_pr14 = pr[5*self.c_in0+5*self.c_in1+5*self.c_in2:5*self.c_in0+5*self.c_in1+5*self.c_in2+self.c_in3]
            c_shortcut_pr14 = pr[5*self.c_in0+5*self.c_in1+5*self.c_in2+self.c_in3:5*self.c_in0+5*self.c_in1+5*self.c_in2+2*self.c_in3]
            c_pr15 = pr[5*self.c_in0+5*self.c_in1+5*self.c_in2+2*self.c_in3:5*self.c_in0+5*self.c_in1+5*self.c_in2+3*self.c_in3]
            c_pr16 = pr[5*self.c_in0+5*self.c_in1+5*self.c_in2+3*self.c_in3:5*self.c_in0+5*self.c_in1+5*self.c_in2+4*self.c_in3]
            c_pr17 = pr[5*self.c_in0+5*self.c_in1+5*self.c_in2+4*self.c_in3:5*self.c_in0+5*self.c_in1+5*self.c_in2+5*self.c_in3]
            f_pr18 = pr[5*self.c_in0+5*self.c_in1+5*self.c_in2+5*self.c_in3:5*self.c_in0+5*self.c_in1+5*self.c_in2+5*self.c_in3+self.f_in]
            # 1~17
            out = self.c_layer1(input, c_pr1, 1)
            out, out0 = self.c_layer2(out, c_pr2, 1)
            out = self.c_layer3(out, c_pr3, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, c_pr4, 1)
            out = self.c_layer5(out, c_pr5, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer6(out, c_pr6, c_shortcut_pr6, 1)
            out = self.c_layer7(out, c_pr7, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, c_pr8, 1)
            out = self.c_layer9(out, c_pr9, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer10(out, c_pr10, c_shortcut_pr10, 1)
            out = self.c_layer11(out, c_pr11, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, c_pr12, 1)
            out = self.c_layer13(out, c_pr13, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer14(out, c_pr14, c_shortcut_pr14, 1)
            out = self.c_layer15(out, c_pr15, 1)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, c_pr16, 1)
            out = self.c_layer17(out, c_pr17, 1)
            out += out0
            out = self.relu(out)
            out = self.avgpool(out)
            x = torch.flatten(out, 1)
            # 18
            out = self.f_layer18(x, f_pr18, 1)
            return out


class Artificial_Astrocyte_Network(nn.Module):
    def __init__(self,):
        super(Artificial_Astrocyte_Network, self).__init__()
        self.dw_weights0 = Parameter(torch.Tensor(16, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(16)
        # self.dw_bias0 = Parameter(torch.Tensor(16))
        self.dw_weights1 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(32)
        # self.dw_bias1 = Parameter(torch.Tensor(32))
        self.dw_weights2 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(64)
        # self.dw_bias2 = Parameter(torch.Tensor(64))
        self.dw_weights3 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(128)
        # self.dw_bias3 = Parameter(torch.Tensor(128))
        self.up_sample0 = Parameter(torch.Tensor(128, 64, 2, 2))
        self.up_bn00 = nn.BatchNorm2d(64)
        # self.up_bias00 = Parameter(torch.Tensor(64))
        self.up_weights0 = Parameter(torch.Tensor(64, 128, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(64)
        # self.up_bias01 = Parameter(torch.Tensor(64))
        self.up_sample1 = Parameter(torch.Tensor(64, 32, 3, 2))
        self.up_bn10 = nn.BatchNorm2d(32)
        # self.up_bias10 = Parameter(torch.Tensor(32))
        self.up_weights1 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(32)
        # self.up_bias11 = Parameter(torch.Tensor(32))
        self.up_sample2 = Parameter(torch.Tensor(32, 16, 3, 2))
        self.up_bn20 = nn.BatchNorm2d(16)
        # self.up_bias20 = Parameter(torch.Tensor(16))
        self.up_weights2 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(16)
        # self.up_bias21 = Parameter(torch.Tensor(16))
        self.gate_weights = Parameter(torch.Tensor(1, 16, 3, 3))
        self.gate_bn = nn.BatchNorm2d(1)
        # self.gate_bias = Parameter(torch.Tensor(1))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.dw_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dw_bn0.weight, 1)
        init.constant_(self.dw_bn1.weight, 1)
        init.constant_(self.dw_bn2.weight, 1)
        init.constant_(self.dw_bn3.weight, 1)
        init.constant_(self.up_bn00.weight, 1)
        init.constant_(self.up_bn01.weight, 1)
        init.constant_(self.up_bn10.weight, 1)
        init.constant_(self.up_bn11.weight, 1)
        init.constant_(self.up_bn20.weight, 1)
        init.constant_(self.up_bn21.weight, 1)
        init.constant_(self.gate_bn.weight, 1)
        # init.constant_(self.dw_bias0, 0)
        # init.constant_(self.dw_bias1, 0)
        # init.constant_(self.dw_bias2, 0)
        # init.constant_(self.dw_bias3, 0)
        # init.constant_(self.up_bias00, 0)
        # init.constant_(self.up_bias01, 0)
        # init.constant_(self.up_bias10, 0)
        # init.constant_(self.up_bias11, 0)
        # init.constant_(self.up_bias20, 0)
        # init.constant_(self.up_bias21, 0)
        # init.constant_(self.gate_bias, 0)

    def forward(self, weights):
        with torch.no_grad():
            layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(weights, self.dw_weights0, stride=1, padding=1, bias=None)))
            layer01 = self.maxpool(layer00)
            layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
            layer11 = self.maxpool(layer10)
            layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
            layer21 = self.maxpool(layer20)
            layer3 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))  # [1, 16, 9, 8]
            layer40 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer3, self.up_sample0, stride=2, bias=None)))
            layer41 = torch.cat((layer20, layer40), dim=1)
            layer42 = F.relu(self.up_bn01(nn.functional.conv2d(layer41, self.up_weights0, stride=1, padding=1, bias=None)))
            layer50 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer42, self.up_sample1, stride=2, bias=None)))
            layer51 = torch.cat((layer10, layer50), dim=1)
            layer52 = F.relu(self.up_bn11(nn.functional.conv2d(layer51, self.up_weights1, stride=1, padding=1, bias=None)))
            layer60 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer52, self.up_sample2, stride=2, bias=None)))
            layer61 = torch.cat((layer00, layer60), dim=1)
            layer62 = F.relu(self.up_bn21(nn.functional.conv2d(layer61, self.up_weights2, stride=1, padding=1, bias=None)))
            layer_out = F.sigmoid(self.gate_bn(nn.functional.conv2d(layer62, self.gate_weights, stride=1, padding=1, bias=None)))  # [1, 1, 83, 64]
            gates = layer_out.view(-1)
            return gates