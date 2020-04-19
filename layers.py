import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):

        super(Conv2d, self).__init__()
        """
        An implementation of a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Parameters:
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - kernel_size: Size of the convolving kernel
        - stride: The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - padding: The number of pixels that will be used to zero-pad the input.
        """


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_H, self.kernel_size_W = kernel_size
        self.weight = torch.randn(out_channels, in_channels, self.kernel_size_H,
                                  self.kernel_size_W, requires_grad = True)
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if bias == True:
            self.bias_weight = torch.zeros(1, requires_grad=True)


    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """


        batch_size = x.shape[0]

        H_in = x.shape[2]
        W_in = x.shape[3]

        H_out = (H_in - self.kernel_size_H + 2*self.padding) / self.stride + 1
        W_out = (W_in - self.kernel_size_W + 2*self.padding) / self.stride + 1

        H_out, W_out = int(H_out), int(W_out)


        input_unfold = F.unfold(x, kernel_size=self.kernel_size,
                                padding=self.padding,
                                stride=self.stride)
        
        kernel_unfold = F.unfold(self.weight, kernel_size = self.kernel_size, stride=self.stride)

        conv_unfold = input_unfold.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)

        if self.bias == True:
            conv_unfold += self.bias_weight

        out = conv_unfold.view(batch_size, self.out_channels, H_out, W_out)        


        return out

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        """
        An implementation of a max-pooling layer.

        Parameters:
        - kernel_size: the size of the window to take a max over
        """

        self.kernel_size = kernel_size
        self.kernel_size_H, self.kernel_size_W = kernel_size


    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """

        input_unfold = F.unfold(x, kernel_size=self.kernel_size,
                                   stride=self.kernel_size, dilation=1).T
        
        pooled, idx = torch.max(input_unfold, 1, keepdim=True)
        out = F.fold(pooled.reshape(1,-1,1), kernel_size=self.kernel_size,
                             output_size=self.kernel_size)


        return out

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        """
        An implementation of a Linear layer.

        Parameters:
        - weight: the learnable weights of the module of shape (in_channels, out_channels).
        - bias: the learnable bias of the module of shape (out_channels).
        """
        self.weight = torch.randn(out_channels, in_channels,
                                   requires_grad=True)
        self.bias = bias
        if bias==True:
            self.bias_weight = torch.zeros(1, requires_grad=True)


    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, *, H) where * means any number of additional
        dimensions and H = in_channels
        Output:
        - out: Output data of shape (N, *, H') where * means any number of additional
        dimensions and H' = out_channels
        """

        out = x.matmul(self.weight.t())
        if self.bias:
            out += self.bias_weight


        return out

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        """
        An implementation of a Batch Normalization over a mini-batch of 2D inputs.

        The mean and standard-deviation are calculated per-dimension over the
        mini-batches and gamma and beta are learnable parameter vectors of
        size num_features.

        Parameters:
        - num_features: C from an expected input of size (N, C, H, W).
        - eps: a value added to the denominator for numerical stability. Default: 1e-5
        - momentum: momentum - the value used for the running_mean and running_var
        computation. Default: 0.1
        - gamma: the learnable weights of shape (num_features).
        - beta: the learnable bias of the module of shape (num_features).
        """

        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(num_features,
                                 requires_grad = True)
        self.beta = torch.zeros(num_features,
                                requires_grad = True)
        self.n = 0 #number of times batchnorm is called in training

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.zeros(num_features)



    def forward(self, x):
        """
        During training this layer keeps running estimates of its computed mean and
        variance, which are then used for normalization during evaluation.
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data of shape (N, C, H, W) (same shape as input)
        """

        if torch.is_grad_enabled: # training phase
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps) # normalised
            # now update running metrics


            if self.n > 0:            
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * mean
                self.running_var = (1-self.momentum)*self.running_var + self.momentum * var
            else:
                self.running_mean = mean
                self.running_var = var
            self.n += 1

        else: # evaluation phase
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) # normalised
        
        x_shift = self.gamma * x_norm + self.beta # scale and shift
        x = x_shift

        return x