import torch
import torch.nn as nn
from torch.autograd import Function
#import numpy as np
import gauss_psf_cuda


class GaussPSFFunction(Function):
    @staticmethod
    def forward(ctx, input,ksize, weights,weights_hsv,weights_rgb, kernel_size=7):
        with torch.no_grad():
            x = torch.arange(ksize // 2,
                             -ksize // 2,
                             -1).view(ksize, 1).float().repeat(1, kernel_size).cuda()

            y = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(1, kernel_size).float().repeat(kernel_size, 1).cuda()
        #print('at input',weights_rgb)
        outputs, wsum = gauss_psf_cuda.forward(input,weights,weights_hsv,weights_rgb, x, y)
        ctx.save_for_backward(input, outputs, weights,weights_hsv, wsum, x, y)

        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, outputs, weights, wsum, x, y = ctx.saved_variables
        x = -x
        y = -y
        grad_input, grad_weights = gauss_psf_cuda.backward(grad.contiguous(), input, outputs, weights, wsum, x, y)
        return grad_input, grad_weights, None


class GaussPSF(nn.Module):
    def __init__(self, kernel_size, near=1, far=10, pixel_size=5.6e-6, scale=3):
        super(GaussPSF, self).__init__()
        self.kernel_size = kernel_size
        self.near = near
        self.far = far
        self.pixel_size = pixel_size
        self.scale = scale

    def forward(self, image, depth, focal_depth, apature,ksize, focal_length,weights_hsv,weights_rgb):

        Ap = apature.view(-1, 1, 1).expand_as(depth)
        FL = focal_length.view(-1, 1, 1).expand_as(depth)
        #ksize = ksize.view(-1, 1, 1).expand_as(depth)
        focal_depth = focal_depth.view(-1, 1, 1).expand_as(depth)
        #Ap = FL / Ap

        real_depth =  depth 

        real_fdepth = focal_depth
        

        c = torch.abs(     Ap * (FL * (depth - focal_depth)) / (depth * (focal_depth - FL))              ) 

        c = c.clamp(min=1, max=self.kernel_size)
        #print(c.shape)
        #print(weights_hsv.shape)

        c=c*weights_hsv
        
        #weights_hsv[:,:,:]=1
        #print(weights_hsv)

        weights = c.unsqueeze(1).expand_as(image).contiguous()
        weights_hsv = weights_hsv.unsqueeze(1).expand_as(image).contiguous()
        #print(weights_rgb,weights_rgb.shape)
        weights_rgb = weights_rgb.unsqueeze(1).contiguous()
        #print(weights_rgb,weights_rgb.shape)
        #print(weights.shape)
        #print(weights_hsv.shape)
        return GaussPSFFunction.apply(image,ksize, weights,weights_hsv,weights_rgb, self.kernel_size)
