import torch.nn as nn
import torch
import torch.nn.functional as F


class BER_Mask(torch.autograd.Function):
    ''' Mask the binary activations at a specific BER'''

    @staticmethod
    def forward(ctx, input):
        BER = 0.01
        ctx.save_for_backward(input)
        if BER < 1e-14:
            return input.clone()
        #m = torch.cuda.FloatTensor(input.size())
        #mask = ( torch.rand(m.size(),out=m) > error).float()
        mask = ( torch.rand(*input.size()) >  BER).float()
        mask *= 2
        mask -= 1
        return input*mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class BinActiveZ(torch.autograd.Function):
    ''' Binarize the input activations'''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_bin = grad_output.clone()
        grad_bin[input.ge(1)] = 0
        grad_bin[input.le(-1)] = 0
        return grad_bin  


class BinConv2d(nn.Module):
    ''' Binary Convolution composite layer '''

    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.BinaryActivation = BinActiveZ.apply
        self.ActivationErrors = BER_Mask.apply


        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, testing = False):
        x = self.bn(x)
        x = self.BinaryActivation(x)
        if not testing:
            x = self.ActivationErrors(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class NiN(nn.Module):
    ''' Binarized Network in Network '''

    def __init__(self):
        super(NiN, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)

        self.bin_conv2 = BinConv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.bin_conv3 = BinConv2d(160,  96, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bin_conv4 = BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5)
        self.bin_conv5 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bin_conv6 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.av_conv6 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bin_conv7 = BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.bin_conv8 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn_conv8 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)

        self.conv9 = nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0)
        self.relu_conv9 = nn.ReLU(inplace=True)
        self.av_conv9 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)


    def forward(self, x, testing = False):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)

        x = self.bin_conv2(x, testing)
        x = self.bin_conv3(x, testing)
        x = self.pool3(x)

        x = self.bin_conv4(x, testing)
        x = self.bin_conv5(x, testing)
        x = self.bin_conv6(x, testing)
        x = self.av_conv6(x)

        x = self.bin_conv7(x, testing)
        x = self.bin_conv8(x, testing)
        x = self.bn_conv8(x)

        x = self.conv9(x)
        x = self.relu_conv9(x)
        x = self.av_conv9(x)

        x = x.view(x.size(0), 10)
        return x
