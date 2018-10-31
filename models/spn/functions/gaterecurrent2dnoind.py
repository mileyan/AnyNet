import torch
from torch.autograd import Function
from .._ext import gaterecurrent2dnoind as gaterecurrent2d

class GateRecurrent2dnoindFunction(Function):

    @staticmethod
    def forward(ctx, X, G1, G2, G3, horizontal, reverse):
        num, channels, height, width = X.size()
        output = torch.zeros(num, channels, height, width)

        if not X.is_cuda:
            print("cpu version is not ready at this time")
            return 0
        else:
            output = output.cuda()
            gaterecurrent2d.gaterecurrent2dnoind_forward_cuda(horizontal,reverse, X, G1, G2, G3, output)
            ctx.save_for_backward(X, G1, G2, G3, output)
            ctx.hiddensize = X.size()
            ctx.horizontal = horizontal
            ctx.reverse = reverse
            return output

    @staticmethod
    def backward(ctx, grad_output):
        assert(ctx.hiddensize is not None and grad_output.is_cuda)
        num, channels, height, width = ctx.hiddensize
        X, G1, G2, G3, output = ctx.saved_tensors

        grad_X = torch.zeros(num, channels, height, width).cuda()
        grad_G1 = torch.zeros(num, channels, height, width).cuda()
        grad_G2 = torch.zeros(num, channels, height, width).cuda()
        grad_G3 = torch.zeros(num, channels, height, width).cuda()

        gaterecurrent2d.gaterecurrent2dnoind_backward_cuda(ctx.horizontal, ctx.reverse, output, grad_output, X, G1, G2, G3, grad_X, grad_G1, grad_G2, grad_G3)

        return (grad_X, grad_G1, grad_G2, grad_G3)+(None,)*2
