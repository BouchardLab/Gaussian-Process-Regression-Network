import torch
import numpy as np
err = 1e-8

dtype = torch.float


def default_postprocess_script(x):
    return x


class Distance(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            x2_norm, x2_pad = x1_norm, x1_pad
        else:
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
        res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if postprocess else res


class Kernel_RBF(torch.nn.Module):
    def __init__(self, raw_lengthscale=0., raw_scale=0., require_grad_lengthscale=True, require_grad_scale=True):
        super().__init__()
        super().register_parameter(name="raw_lengthscale", param=torch.nn.Parameter(torch.tensor(raw_lengthscale)))
        super().register_parameter(name="raw_scale", param=torch.nn.Parameter(torch.tensor(raw_scale)))
        self.raw_lengthscale.requires_grad = require_grad_lengthscale
        self.raw_scale.requires_grad = require_grad_scale

    def forward(self, x1, x2):
        # self.scale = torch.exp(self.raw_scale)
        self.scale = torch.exp(self.raw_scale)
        self.lengthscale = torch.exp(self.raw_lengthscale)
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        res = self.scale * self.RBF(x1_, x2_)
        return res

    def covar_dist(self, x1, x2, dist_postprocess_func=default_postprocess_script,
                   postprocess=True):
        self.distance_module = Distance(dist_postprocess_func)
        x1_eq_x2 = torch.equal(x1, x2)
        res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
        return res

    def postprocess_rbf(self, dist_mat):
        return dist_mat.div_(-2).exp_()

    def RBF(self, x1, x2):
        return self.covar_dist(x1, x2, dist_postprocess_func=self.postprocess_rbf, postprocess=True)


def psd_inv(K, device='cpu'):
    while True:
        try:
            u = torch.cholesky(K)
            break
        except:
            K += err * torch.eye(K.shape[0]).to(device)
    K_inv = torch.cholesky_inverse(u)
    return K_inv


def psd_cholesky(K, device='cpu'):
    while True:
        try:
            u = torch.cholesky(K)
            break
        except:
            K += err * torch.eye(K.shape[0]).to(device)
    return u