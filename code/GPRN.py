# Reproduce GPRN model

import numpy as np
import torch
import math
from scipy.special import logsumexp
from tqdm import tqdm
import utils
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GPRN_model(torch.nn.Module):
    def __init__(self, kernel, P, Q, N, device='cpu', dtype=torch.float64):
        super().__init__()
        print('Initialize GPRN model')
        self.P = P
        self.Q = Q
        self.N = N
        self.device = device
        self.dtype = dtype

        # f, w kernel
        if kernel == 'rbf':
            self.kernel_f = utils.Kernel_RBF().to(self.device)
            self.kernel_w = utils.Kernel_RBF().to(self.device)
        else:
            raise ValueError

        # noise variance
        self.raw_sigma2_f = torch.nn.Parameter(torch.tensor(0., dtype=self.dtype, device=self.device))
        self.raw_sigma2_y = torch.nn.Parameter(torch.tensor(0., dtype=self.dtype, device=self.device))

        # intialize variational parameters
        self.f_mu = torch.nn.Parameter(torch.randn(self.Q, self.N, dtype=self.dtype, device=self.device))
        self.f_raw_sigma2 = torch.nn.Parameter(torch.zeros(self.Q, self.N, dtype=self.dtype, device=self.device))
        self.w_mu = torch.nn.Parameter(torch.randn(self.P, self.Q, self.N, dtype=self.dtype, device=self.device))
        self.w_raw_sigma2 = torch.nn.Parameter(
            torch.randn(self.P, self.Q, self.N, dtype=self.dtype, device=self.device))

    def loss(self, X, Y):
        sigma2_y = torch.exp(self.raw_sigma2_y)
        sigma2_f = torch.exp(self.raw_sigma2_f)
        f_sigma2 = torch.exp(self.f_raw_sigma2) # Q x N
        w_sigma2 = torch.exp(self.w_raw_sigma2) # P x Q x N
        f_mu = self.f_mu.transpose(1,0) # N x Q
        f_mu_expand = f_mu[..., None] # N x Q x 1
        w_mu = self.w_mu.permute(2,0,1) # N x P x Q
        # compute expectation of conditional log likelihood
        diff = Y - torch.matmul(w_mu, f_mu_expand).squeeze()           #shape: N x P
        w_mu2 = w_mu ** 2 # N x P x Q
        f_mu2 = f_mu_expand ** 2 # N x Q x 1
        f_sigma2_expand = torch.unsqueeze(f_sigma2, 0).expand(self.P, -1, -1)     # shape P x Q x N
        f_mu2_expand = f_mu2.expand(-1, -1, self.P) # shape N x Q x P
        exp_llik = -0.5*self.N*self.P*torch.log(2*np.pi*sigma2_y) - 0.5/sigma2_y*torch.sum(diff**2) - \
                   0.5/sigma2_y*((w_mu2.permute(1,2,0)*f_sigma2_expand).sum() + (w_sigma2*f_mu2_expand.permute(2,1,0)).sum())
        # compute expectation of log joint probability of latent variables
        K_f = self.kernel_f(X, X) + sigma2_f*torch.eye(self.N).to(self.device)
        chol_K_f = utils.psd_cholesky(K_f, device=self.device)
        inv_K_f = utils.psd_inv(K_f, device=self.device)
        scaled_f_mu = torch.triangular_solve(f_mu, chol_K_f, upper=False)[0]
        K_w = self.kernel_w(X, X)
        chol_K_w = utils.psd_cholesky(K_w, device=self.device)
        inv_K_w = utils.psd_inv(K_w, device=self.device)
        scaled_w_mu = torch.triangular_solve(self.w_mu[..., None], chol_K_w, upper=False)[0].squeeze() # P x Q x N
        exp_jlp = -0.5*(self.Q*torch.logdet(K_f) + torch.sum(scaled_f_mu**2) + torch.matmul(f_sigma2,
                    torch.diag(inv_K_f)).sum()) - 0.5 * (self.P*self.Q*torch.logdet(K_w) + torch.sum(scaled_w_mu**2) +
                    torch.matmul(w_sigma2, torch.diag(inv_K_w)).sum())
        # compute entropies
        H = 0.5*self.f_raw_sigma2.sum() + 0.5*self.w_raw_sigma2.sum()

        elbo = exp_llik + exp_jlp + H

        # print(exp_llik, exp_jlp, H)

        return -elbo

    def predict(self, X_train, X):
        sigma2_f = torch.exp(self.raw_sigma2_f)

        Kw11 = self.kernel_w(X_train, X_train) # N x N
        Kw12 = self.kernel_w(X_train, X) # N x M
        Kw11InvKw12 = torch.solve(A=Kw11, input=Kw12)[0] # N x M
        W_p = torch.matmul(torch.unsqueeze(self.w_mu, dim=-2), Kw11InvKw12[None,None,...]).squeeze() # P x Q x M
        Kf11 = self.kernel_f(X_train, X_train) + sigma2_f*torch.eye(self.N).to(self.device) # N x N
        Kf12 = self.kernel_f(X_train, X)  # N x M
        Kf11InvKf12 = torch.solve(A=Kf11, input=Kf12)[0]  # N x M
        f_p = torch.matmul(torch.unsqueeze(self.f_mu, dim=-2), Kf11InvKf12[None,None,...]).squeeze() # Q x M
        Y_p = torch.matmul(W_p.permute(2,0,1), f_p.permute(1,0)[..., None]).squeeze() # M P
        return Y_p


class GPRN(torch.nn.Module):
    def __init__(self, config, device='cpu', dtype=torch.float64):
        super().__init__()
        print('Initialize GPRN')

        self.config = config
        self.Q = self.config['Q']
        self.in_d = self.config['data']['X_train'].shape[1]
        self.out_d = self.config['data']['Y_train'].shape[1]
        self.jitter = config['jitter']
        self.kernel = self.config['kernel']
        self.device = device
        self.dtype=dtype

        self.X_train = self.config['data']['X_train'].reshape((-1, self.in_d))
        self.X_test = self.config['data']['X_test'].reshape((-1, self.in_d))
        self.Y_train = self.config['data']['Y_train'].reshape((-1, self.out_d))
        self.Y_test = self.config['data']['Y_test'].reshape((-1, self.out_d))
        self.Y_test_ground = self.config['data']['Y_test_ground'].reshape((-1, self.out_d))
        self.Y_mean = self.config['data']['Y_mean']
        self.Y_std = self.config['data']['Y_std']

        self.epochs = config['epochs']
        self.lr = config['lr']

        self.record_time = config['record_time']

        self.M = self.X_test.shape[0]
        self.N = self.X_train.shape[0]
        self.P = self.out_d
        self.model = GPRN_model(self.kernel, self.P, self.Q, self.N, dtype=self.dtype, device=self.device)

    def fit(self, epochs=None):
        if epochs is None:
            epochs = self.epochs
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        res = dict()
        loss_list = list()
        hist_pred = list()
        # hist_nrmse = list()
        # hist_nmae = list()
        X_train = torch.from_numpy(self.X_train).to(self.device)
        Y_train = torch.from_numpy(self.Y_train).to(self.device)
        X_test = torch.from_numpy(self.X_test).to(self.device)
        Y_test_ground = torch.from_numpy(self.Y_test_ground).to(self.device)
        Y_mean = torch.from_numpy(self.Y_mean).to(self.device)
        Y_std = torch.from_numpy(self.Y_std).to(self.device)

        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.model.loss(X_train, Y_train)
            loss.backward()
            optimizer.step()
            pred_Y = self.model.predict(X_train, X_test) * Y_std + Y_mean
            # r0 = torch.sqrt(torch.mean((pred_Y - Y_test_ground)**2)) / torch.std(Y_test_ground)
            # r1 = torch.mean(torch.abss(pred_Y - Y_test_ground)) / torch.std(Y_test_ground)
            if epoch % 100 == 0:
                print("epoch {}, loss: {}.".format(epoch, loss))
            loss_list.append(loss.item())
            hist_pred.append(pred_Y.to('cpu').detach().numpy())

        res['loss'] = loss_list
        res['Y_pred'] = hist_pred
        # breakpoint()
        return res

