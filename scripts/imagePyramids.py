
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
            x_out = [f(x).to(torch.bool) for f in self.multiscales]
        else:
            x_out = [f(x) for f in self.multiscales]
        return x_out

def downsample(x, scale, mode):
    if scale == 1.0:
        return x
    _, _, H, W = x.shape
    new_H, new_W = int(H * scale), int(W * scale)
    return torch.nn.functional.interpolate(x, size=(new_H, new_W), mode=mode, align_corners=True)