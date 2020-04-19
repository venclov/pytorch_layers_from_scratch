from layers import Conv2d, MaxPool2d, BatchNorm2d, Linear
import torch.nn as nn
import torch
import torch.nn.functional as F

def test_conv_without_bias():
    input_matrix  = torch.Tensor([[[[3., 9., 0.],
                                  [2., 8., 1.],
                                  [1., 4., 8.]]]])

    kernel = torch.Tensor([[[[8., 9.],
                          [4., 4.]]]])
    
    torch_conv = nn.Conv2d(1, 1, 2, stride=1, bias=False)
    torch_conv.weight.data = kernel

    paul_conv = Conv2d(1,1,(2,2),bias=False)
    paul_conv.weight = kernel


    res_torch = torch_conv(input_matrix)
    res_paul = paul_conv(input_matrix)

    assert torch.allclose(res_torch, res_paul)


def test_conv_with_bias():
    input_matrix  = torch.Tensor([[[[3., 9., 0.],
                                  [2., 8., 1.],
                                  [1., 4., 8.]]]])

    kernel = torch.Tensor([[[[8., 9.],
                          [4., 4.]]]])

                        
    bias = torch.Tensor([1.])
    
    torch_conv = nn.Conv2d(1, 1, 2, stride=1, bias=True)
    torch_conv.weight.data = kernel
    torch_conv.bias.data = bias
    torch_conv_op = torch_conv(input_matrix)

    paul_conv = Conv2d(1,1,(2,2),bias=True)
    paul_conv.weight = kernel

    paul_conv.bias_weight = bias

    res_torch = torch_conv_op
    res_paul = paul_conv(input_matrix)

    assert torch.allclose(res_torch, res_paul)


def test_conv_with_bias_zeros():
    input_matrix  = torch.Tensor([[[[3., 9., 0.],
                                  [2., 8., 1.],
                                  [1., 4., 8.]]]])

    kernel = torch.Tensor([[[[8., 9.],
                          [4., 4.]]]])

                        
    bias = torch.Tensor([1.])
    
    torch_conv = nn.Conv2d(1, 1, 2, stride=1, bias=True)
    torch_conv.weight.data = kernel
    torch_conv.bias.data = bias
    torch_conv_op = torch_conv(input_matrix)

    paul_conv = Conv2d(1,1,(2,2) ,bias=True)
    paul_conv.weight = kernel
    
    paul_conv.bias_weight = bias

    res_torch = torch_conv_op
    res_paul = paul_conv(input_matrix)

    assert torch.allclose(res_torch, res_paul)


def test_linear():
    input_matrix  = torch.Tensor([[[3., 9., 0.]]])
                    
    weights = torch.Tensor([[1., 1., 1.]])
    
    torch_linear = nn.Linear(1, 1, bias=False)
    torch_linear.weight.data = weights

    paul_linear = Linear(1,1, bias=False)
    paul_linear.weight = weights
    

    res_torch = torch_linear(input_matrix)
    res_paul = paul_linear(input_matrix)

    assert torch.allclose(res_torch, res_paul)


def test_linear_bias():
    input_matrix  = torch.Tensor([[[[3., 9., 0.],
                                [2., 8., 1.],
                                [1., 4., 8.]]]])
                    

    torch_linear = nn.Linear(1,1)

    weights = torch.Tensor([[1., 1., 1.]])
    bias = torch.Tensor([1.])

    torch_linear.weight.data = weights
    torch_linear.bias.data = bias

    paul_linear = Linear(1,1, bias=True)
    paul_linear.weight = weights
    paul_linear.bias_weight = bias
    

    res_torch = torch_linear(input_matrix)
    res_paul = paul_linear(input_matrix)

    assert torch.allclose(res_torch, res_paul)

def test_maxpool():
    input_matrix  = torch.Tensor([[[[1., 2., 3., 4.],
                                  [5., 6., 7., 8.],
                                  [9., 10., 11., 12.],
                                  [13., 14., 15., 16.]]]])
    
    torch_conv = nn.MaxPool2d(2, stride=2)
    paul_conv = MaxPool2d((2,2))

    res_torch = torch_conv(input_matrix)
    res_paul = paul_conv(input_matrix)

    assert torch.allclose(res_torch, res_paul)


def test_batch_norm():
    input_matrix  = torch.Tensor([[[[1., 1., 1.],[2., 2., 2.]]]])
                    
    
    torch_BatchNorm2d = nn.BatchNorm2d(1)
    paul_BatchNorm2d = BatchNorm2d(1)
    
    res_torch = torch_BatchNorm2d(input_matrix)
    res_paul = paul_BatchNorm2d(input_matrix)

    assert torch.allclose(res_torch, res_paul)


def test_conv_without_bias_rand():
    input_matrix = torch.rand(1,1,3,3)

    kernel = torch.Tensor([[[[8., 9.],
                          [4., 4.]]]])
    
    torch_conv = nn.Conv2d(1, 1, 2, stride=1, bias=False)
    torch_conv.weight.data = kernel

    paul_conv = Conv2d(1,1,(2,2),bias=False)
    paul_conv.weight = kernel


    res_torch = torch_conv(input_matrix)
    res_paul = paul_conv(input_matrix)

    assert torch.allclose(res_torch, res_paul)
    
def test_conv_without_bias_rand_channels():
    in_channels = 2
    input_matrix = torch.rand(2,in_channels,3,3)

    kernel = torch.rand(1,2,2,2)
    
    torch_conv = nn.Conv2d(in_channels, 1, 2, stride=1, bias=False)
    torch_conv.weight.data = kernel

    paul_conv = Conv2d(in_channels, 1, (2,2) ,bias=False)
    paul_conv.weight = kernel


    res_torch = torch_conv(input_matrix)
    res_paul = paul_conv(input_matrix)

    assert torch.allclose(res_torch, res_paul)