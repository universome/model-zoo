import torch
import torch.nn as nn


class Dropword(nn.Module):
    def __init__(self, p):
        super(Dropword, self).__init__()
        self.p = p

    def forward(self, x, p:float=None):
        assert x.dim() == 3 # (batch, len, emb_size)

        p = p or self.p
        mask = torch.bernoulli(torch.Tensor(x.size(0), x.size(1)).fill_(1 - p))
        mask = mask.to(x.device)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(2))

        return x * mask if self.training else x


class NoiseLayer(nn.Module):
    def __init__(self, sigma):
        super(NoiseLayer, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if not self.training: return x

        noise = torch.zeros_like(x).normal_()

        return x + self.sigma * noise


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    Copypasted from https://github.com/akanimax/pro_gan_pytorch
    """

    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], dim=1)

        # return the computed values:
        return y
