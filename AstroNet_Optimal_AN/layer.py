import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Convolution1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None)))
            return output


#  First Residual Block
class Convolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, input, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None)))
            return output, input


class Convolution3(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


class Convolution4(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, input, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None)))
            return output, input


class Convolution5(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution5, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


#  Second Residual Block
class Convolution6(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution6, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, input, pr1, pr2, shortcut_pr1, shortcut_pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            shortcut_weight_mean = torch.mean(self.shortcut_weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean, shortcut_weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input,  self.weights_new, stride=2, padding=1, bias=None)))
            weight_max, id = torch.max( self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            shortcut_pr1 = shortcut_pr1.view(self.out_features, 1, 1, 1)
            self.shortcut_weights_new = self.shortcut_weights.mul(shortcut_pr1)
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, self.shortcut_weights_new, stride=2, padding=1, bias=None))
            shortcut_weight_max, id = torch.max(self.shortcut_weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, shortcut_output, weight_max, shortcut_weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=2, padding=1, bias=None)))
            shortcut_pr2 = shortcut_pr2.view(self.out_features, 1, 1, 1)
            shortcut_weight_finial = self.shortcut_weights_new.mul(shortcut_pr2)
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, shortcut_weight_finial, stride=2, padding=1, bias=None))
            return output, shortcut_output


class Convolution7(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution7, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


class Convolution8(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution8, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, input, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None)))
            return output, input


class Convolution9(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution9, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


#  Third Residual Block
class Convolution10(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution10, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, input, pr1, pr2, shortcut_pr1, shortcut_pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            shortcut_weight_mean = torch.mean(self.shortcut_weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean, shortcut_weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=2, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            shortcut_pr1 = shortcut_pr1.view(self.out_features, 1, 1, 1)
            self.shortcut_weights_new = self.shortcut_weights.mul(shortcut_pr1)
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, self.shortcut_weights_new, stride=2, padding=1, bias=None))
            shortcut_weight_max, id = torch.max(self.shortcut_weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, shortcut_output, weight_max, shortcut_weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=2, padding=1, bias=None)))
            shortcut_pr2 = shortcut_pr2.view(self.out_features, 1, 1, 1)
            shortcut_weight_finial = self.shortcut_weights_new.mul(shortcut_pr2)
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, shortcut_weight_finial, stride=2, padding=1, bias=None))
            return output, shortcut_output


class Convolution11(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution11, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


class Convolution12(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution12, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, input, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None)))
            return output, input


class Convolution13(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution13, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


#  Forth Residual Block
class Convolution14(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution14, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, input, pr1, pr2, shortcut_pr1, shortcut_pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            shortcut_weight_mean = torch.mean(self.shortcut_weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean, shortcut_weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=2, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            shortcut_pr1 = shortcut_pr1.view(self.out_features, 1, 1, 1)
            self.shortcut_weights_new = self.shortcut_weights.mul(shortcut_pr1)
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, self.shortcut_weights_new, stride=2, padding=1, bias=None))
            shortcut_weight_max, id = torch.max(self.shortcut_weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, shortcut_output, weight_max, shortcut_weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=2, padding=1, bias=None)))
            shortcut_pr2 = shortcut_pr2.view(self.out_features, 1, 1, 1)
            shortcut_weight_finial = self.shortcut_weights_new.mul(shortcut_pr2)
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, shortcut_weight_finial, stride=2, padding=1, bias=None))
            return output, shortcut_output


class Convolution15(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution15, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


class Convolution16(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None)))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, input, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = F.relu(self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None)))
            return output, input


class Convolution17(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution17, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            weight_mean = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return weight_mean
        elif i == 1:
            pr1 = pr1.view(self.out_features, 1, 1, 1)
            self.weights_new = self.weights.mul(pr1)
            output = self.bn(nn.functional.conv2d(input, self.weights_new, stride=1, padding=1, bias=None))
            weight_max, id = torch.max(self.weights_new.view(self.out_features, self.in_features * 3 * 3), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.out_features, 1, 1, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = self.bn(nn.functional.conv2d(input, weight_finial, stride=1, padding=1, bias=None))
            return output


class Fully_Connection(nn.Module):
    def __init__(self, in_features, out_features):
        super(Fully_Connection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        # self.bias = Parameter(torch.Tensor(out_features))
        self.weights_new = torch.Tensor(out_features, in_features, 3, 3)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weights, 0, 0.01)
        # init.constant_(self.bias, 0)

    def forward(self, input, pr1, pr2, i):
        if i == 0:
            data = torch.mean(self.weights.view(self.in_features, self.out_features), dim=1)
            return data
        elif i == 1:
            pr1 = pr1.view(self.in_features, 1)
            self.weights_new = self.weights.mul(pr1)
            output = input.mm(self.weights_new)
            weight_max, id = torch.max(self.weights_new.view(self.in_features, self.out_features), dim=1)
            return output, weight_max
        else:
            pr2 = pr2.view(self.in_features, 1)
            weight_finial = self.weights_new.mul(pr2)
            output = input.mm(weight_finial)
            # output.add_(self.bias)
            return output