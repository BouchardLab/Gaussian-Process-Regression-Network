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
    def __init__(self, kernel, P, Q, N_list, device='cpu', dtype=torch.float64):
        super().__init__()
        print('Initialize GPRN model')
        self.P = P
        self.Q = Q
        self.N_list = N_list
        self.device = device
        self.dtype = dtype

        # f, w kernel
        if kernel == 'rbf':
            self.kernel_f = utils.Kernel_RBF().to(self.device)
            self.kernel_w = utils.Kernel_RBF().to(self.device)
        else:
            raise ValueError

        # noise variance
        self.raw_sigma2_f = torch.nn.Parameter(torch.tensor(-3., dtype=self.dtype, device=self.device))
        self.raw_sigma2_w = torch.nn.Parameter(torch.tensor(-3., dtype=self.dtype, device=self.device))
        self.raw_sigma2_y = torch.nn.Parameter(torch.tensor(-3., dtype=self.dtype, device=self.device))

        # intialize variational parameters
        self.f_mu = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.Q, N, dtype=self.dtype, device=self.device)) for N in self.N_list]) # [Q, N_P]
        self.f_raw_sigma2 = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.Q, N, dtype=self.dtype, device=self.device)) for N in self.N_list]) # [Q, N_P]
        self.w_mu = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.P, self.Q, N, dtype=self.dtype, device=self.device)) for N in self.N_list]) # [P, Q, N_P]
        self.w_raw_sigma2 = torch.nn.ParameterList([torch.nn.Parameter(
            torch.randn(self.P, self.Q, N, dtype=self.dtype, device=self.device)) for N in self.N_list]) # [P, Q, N_P]

    def loss(self, X, Y):
        sigma2_y = torch.exp(self.raw_sigma2_y)
        sigma2_f = torch.exp(self.raw_sigma2_f)
        sigma2_w = torch.exp(self.raw_sigma2_w)
        f_sigma2 = [torch.exp(f_raw_sigma2) for f_raw_sigma2 in self.f_raw_sigma2] # [Q, N_P]
        w_sigma2 = [torch.exp(w_raw_sigma2) for w_raw_sigma2 in self.w_raw_sigma2] # [P x Q x N_P]
        f_mu = [f_mu.transpose(1,0) for f_mu in self.f_mu] # [N_P x Q]
        f_mu_expand = [f_mu_p[..., None] for f_mu_p in f_mu] # [N_P x Q x 1]
        w_mu = [w_mu_p.permute(2,0,1) for w_mu_p in self.w_mu] # [N_P x P x Q]
        # compute expectation of conditional log likelihood
        diff = [Y_p.reshape(-1) - torch.matmul(w_mu_p, f_mu_expand_p).squeeze()[..., index] for index, Y_p, w_mu_p, f_mu_expand_p in zip(range(self.P), Y, w_mu, f_mu_expand)]  # [N_P]
        w_mu2 = [w_mu_p ** 2 for w_mu_p in w_mu] # [N_P x P x Q]
        f_mu2 = [f_mu_p ** 2 for f_mu_p in f_mu] # [N_P x Q]
        exp_llik = torch.stack([-0.5*N_p*torch.log(2*np.pi*sigma2_y) - 0.5/sigma2_y*torch.sum(diff_p**2) - \
                   0.5/sigma2_y*((w_mu2_p.permute(1,2,0)[index]*f_sigma2_p).sum() + (w_sigma2_p.permute(1,2,0)[index]*f_mu2_p).sum())
                   for index, N_p, diff_p, w_mu2_p, w_sigma2_p, f_sigma2_p, f_mu2_p in zip(range(self.P), self.N_list, diff, w_mu2, w_sigma2,
                   f_sigma2, f_mu2)]).sum()

        # print ([torch.sqrt(torch.mean(diff_p**2)) for diff_p in diff])
        # breakpoint()

        # compute expectation of log joint probability of latent variables
        X_all = torch.cat(X)
        f_mu_all = torch.cat(f_mu)
        f_sigma2_all = torch.cat(f_sigma2, axis=-1)
        w_mu_all = torch.cat(w_mu)
        w_sigma2_all = torch.cat(w_sigma2, axis=-1)
        K_f = self.kernel_f(X_all, X_all) + sigma2_f*torch.eye(sum(self.N_list)).to(self.device)
        chol_K_f = utils.psd_cholesky(K_f, device=self.device)
        inv_K_f = utils.psd_inv(K_f, device=self.device)
        scaled_f_mu = torch.triangular_solve(f_mu_all, chol_K_f, upper=False)[0]
        K_w = self.kernel_w(X_all, X_all) + sigma2_w*torch.eye(sum(self.N_list)).to(self.device)
        chol_K_w = utils.psd_cholesky(K_w, device=self.device)
        inv_K_w = utils.psd_inv(K_w, device=self.device)
        scaled_w_mu = torch.triangular_solve(w_mu_all.permute(1,2,0)[..., None], chol_K_w, upper=False)[0].squeeze()  # P x Q x N
        exp_jlp = -0.5*(self.Q*torch.logdet(K_f) + torch.sum(scaled_f_mu**2) + torch.matmul(f_sigma2_all, torch.diag(inv_K_f)).sum()) - \
                  0.5 * (self.P*self.Q*torch.logdet(K_w) + torch.sum(scaled_w_mu**2) + torch.matmul(w_sigma2_all, torch.diag(inv_K_w)).sum())
        # breakpoint()

        # compute entropies
        f_raw_sigma2 = [f_raw_sigma2_p for f_raw_sigma2_p in self.f_raw_sigma2]
        w_raw_sigma2 = [w_raw_sigma2_p for w_raw_sigma2_p in self.w_raw_sigma2]
        H = 0.5* torch.cat(f_raw_sigma2, axis=-1).sum() + 0.5*torch.cat(w_raw_sigma2, axis=-1).sum()

        elbo = exp_llik + exp_jlp + H
        # breakpoint()

        return -elbo

    def predict(self, X_train, X):
        sigma2_f = torch.exp(self.raw_sigma2_f)
        sigma2_w = torch.exp(self.raw_sigma2_w)
        w_mu = [w_mu_p for w_mu_p in self.w_mu]
        f_mu = [f_mu_p for f_mu_p in self.f_mu]

        X_train_all = torch.cat(X_train)
        X_test_all = torch.cat(X)
        Kw11 = self.kernel_w(X_train_all, X_train_all) + sigma2_w*torch.eye(sum(self.N_list)).to(self.device)# N x N
        Kw12 = self.kernel_w(X_train_all, X_test_all) # N x M
        Kw11InvKw12 = torch.solve(A=Kw11, input=Kw12)[0] # N x M
        W_p = torch.matmul(torch.unsqueeze(torch.cat(w_mu, axis=-1), dim=-2), Kw11InvKw12[None,None,...]).squeeze() # P x Q x M
        Kf11 = self.kernel_f(X_train_all, X_train_all) + sigma2_f*torch.eye(sum(self.N_list)).to(self.device) # N x N
        Kf12 = self.kernel_f(X_train_all, X_test_all)  # N x M
        Kf11InvKf12 = torch.solve(A=Kf11, input=Kf12)[0]  # N x M
        f_p = torch.matmul(torch.unsqueeze(torch.cat(f_mu, axis=-1), dim=-2), Kf11InvKf12[None,None,...]).squeeze() # Q x M
        Y_p = torch.matmul(W_p.permute(2,0,1), f_p.permute(1,0)[..., None]).squeeze() # M P
        y_p = list()
        curr_index = 0
        for index, X_p in enumerate(X):
            N_p = X_p.shape[0]
            y_p.append(Y_p[curr_index:curr_index+N_p, index])
            curr_index += N_p
        # breakpoint()
        return y_p


class GPRN(torch.nn.Module):
    def __init__(self, config, device='cpu', dtype=torch.float64):
        super().__init__()
        print('Initialize GPRN')

        self.config = config
        self.Q = self.config['Q']


        self.in_d = self.config['data']['X_train'][0].shape[1]
        self.out_d = len(self.config['data']['Y_train'])
        self.jitter = config['jitter']
        self.kernel = self.config['kernel']
        self.device = device
        self.dtype=dtype

        self.X_train = self.config['data']['X_train']
        self.X_test = self.config['data']['X_test']
        self.X_mean = self.config['data']['X_mean']
        self.X_std = self.config['data']['X_std']
        self.Y_train = self.config['data']['Y_train']
        self.Y_test = self.config['data']['Y_test']
        self.Y_mean = self.config['data']['Y_mean']
        self.Y_std = self.config['data']['Y_std']

        self.epochs = config['epochs']
        self.lr = config['lr']

        self.record_time = config['record_time']

        self.M_list = [X.shape[0] for X in self.X_test]
        self.N_list = [X.shape[0] for X in self.X_train]
        self.P = self.out_d
        self.model = GPRN_model(self.kernel, self.P, self.Q, self.N_list, dtype=self.dtype, device=self.device)

    def fit(self, epochs=None):
        if epochs is None:
            epochs = self.epochs

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        res = dict()
        loss_list = list()
        hist_pred = list()
        X_train = [torch.from_numpy(X).to(self.device) for X in self.X_train]
        Y_train = [torch.from_numpy(Y).to(self.device) for Y in self.Y_train]
        X_test = [torch.from_numpy(X).to(self.device) for X in self.X_test]
        Y_test = [torch.from_numpy(Y).to(self.device) for Y in self.Y_test]
        Y_mean = torch.from_numpy(self.Y_mean).to(self.device)
        Y_std = torch.from_numpy(self.Y_std).to(self.device)
        Y_test_true = [Y*Y_std + Y_mean for Y in Y_test]

        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.model.loss(X_train, Y_train)
            loss.backward()
            optimizer.step()
            pred_Y = [y_p * Y_std + Y_mean for y_p in self.model.predict(X_train, X_test)]
            # breakpoint()
            RMSE = torch.cat([(pred_Y_p.reshape(-1) - Y_test_true_p.reshape(-1))**2 for pred_Y_p, Y_test_true_p in zip(pred_Y, Y_test_true)]).mean().sqrt()

            # r0 = torch.sqrt(torch.mean((pred_Y - Y_test_ground)**2)) / torch.std(Y_test_ground)
            # r1 = torch.mean(torch.abss(pred_Y - Y_test_ground)) / torch.std(Y_test_ground)
            if epoch % 100 == 0:
                print("epoch {}, loss: {}.".format(epoch, loss))
                print("RMSE={}".format(RMSE))
            loss_list.append(loss.item())
            hist_pred.append([y_p.to('cpu').detach().numpy() for y_p in pred_Y])

        res['loss'] = loss_list
        res['Y_pred'] = hist_pred
        # breakpoint()
        return res

