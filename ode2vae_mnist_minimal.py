import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils import data
from torch.distributions import MultivariateNormal, Normal, kl_divergence as kl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchdiffeq import odeint

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from multiprocessing import Process, freeze_support
torch.multiprocessing.set_start_method('spawn', force="True")


# prepare dataset
class Dataset(data.Dataset):
    def __init__(self, Xtr):
        self.Xtr = Xtr # N,16,784
    def __len__(self):
        return len(self.Xtr)
    def __getitem__(self, idx):
        return self.Xtr[idx]
# read data
X = loadmat('rot-mnist-3s.mat')['X'].squeeze() # (N, 16, 784)
N = 500
T = 16
Xtr   = torch.tensor(X[:N],dtype=torch.float32).view([N,T,1,28,28])
Xtest = torch.tensor(X[N:],dtype=torch.float32).view([-1,T,1,28,28])
# Generators
params = {'batch_size': 50, 'shuffle': True, 'num_workers': 2}
training_set = Dataset(Xtr)
training_generator = data.DataLoader(training_set, **params)
test_set = Dataset(Xtest)
test_generator = data.DataLoader(test_set, **params)

# utils
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size//16, 4, 4)


class BNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid_layers: int=2, n_hidden: int=100, logsig0=-4):
        super().__init__()
        self.logsig0    = logsig0
        self.logsig     = Parameter(logsig0*torch.ones(1))
        self.weight_mus = nn.ParameterList([])
        self.bias_mus   = nn.ParameterList([])
        self.acts       = []
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weight_mus.append(Parameter(torch.Tensor(n_out, n_in)))
            self.bias_mus.append(Parameter(torch.Tensor(n_out)))
            self.acts.append(nn.Tanh() if i<n_hid_layers else nn.Identity())
        self.reset_parameters()
        self.sample_weights()

    @property
    def sig(self):
        return torch.exp(self.logsig)

    def reset_parameters(self,gain=1.0):
        for weight in self.weight_mus:
            # nn.init.kaiming_uniform_(weight, a=np.sqrt(5))
            nn.init.xavier_uniform_(weight,gain)
        for i,(weight,bias) in enumerate(zip(self.weight_mus,self.bias_mus)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
        nn.init.uniform_(self.logsig,self.logsig0-1,self.logsig0+1)
        self.sample_weights()

    def _assign_weights(self,x):
        self.weights = [weight+torch.randn_like(weight)*x for weight in self.weight_mus]
        self.biases  = [bias+torch.randn_like(bias)*x for bias in self.bias_mus]

    def sample_weights(self):
        self._assign_weights(self.sig)

    def set_mean_weights(self):
        self._assign_weights(0.0)

    def forward(self, x):
        for (weight,bias,act) in zip(self.weights,self.biases,self.acts):
            x = act(F.linear(x,weight,bias))
        return x

    def __repr__(self):
        str_ = ''
        for i,(weight,act) in enumerate(zip(self.weight_mus,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_


# model implementation
class ODE2VAE(nn.Module):
    def __init__(self, n_chan=1, n_filt=8, q=8):
        super(ODE2VAE, self).__init__()
        h_dim = 512
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_chan, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )
        self.fc1 = nn.Linear(h_dim, 2*q)
        self.fc2 = nn.Linear(h_dim, 2*q)
        self.fc3 = nn.Linear(q, h_dim)
        # differential function
        # self.mlp = nn.Sequential(nn.Linear(2*q,50), nn.Tanh(), nn.Linear(50,50), nn.Tanh(), nn.Linear(50,q))
        self.mlp = BNN(2*q, q, n_hid_layers=2, n_hidden=50)
        self.f = lambda t,x: self.mlp(x)
        # decoder
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )   
        self._zero_mean = torch.zeros(2*q).to(device)
        self._eye_covar = torch.eye(2*q).to(device) 
        self.mvn = MultivariateNormal(self._zero_mean, self._eye_covar)

    def odefp(self,t,vs_logp):
        vs, logp = vs_logp # N,2q & N
        q = vs.shape[1]//2
        dv = self.f(t,vs) # N,q 
        ds = vs[:,:q]  # N,q
        dvs = torch.cat([dv,ds],1) # N,2q
        ddvi_dvi = torch.stack(
                    [torch.autograd.grad(dv[:,i],vs,torch.ones_like(dv[:,i]),
                    retain_graph=True,create_graph=True)[0].contiguous()[:,i]
                    for i in range(q)],1) # N,q --> df(x)_i/dx_i, i=1..q
        tr_ddvi_dvi = torch.sum(ddvi_dvi,1) # N
        return (dvs,-tr_ddvi_dvi)

    def elbo(self, qz_m, qz_logv, zode_L, logpL, X, XrecL, L, qz_enc_m=None, qz_enc_logv=None):
        ''' Input:
                qz_m - latent means [N,2q]
                qz_logv - latent logvars [N,2q]
                zode_L - latent trajectory samples [L,N,T,2q]
                logpL - densities of latent trajectory samples [L,N,T]
                X - input images [N,T,nc,d,d]
                XrecL - reconstructions [L,N,T,nc,d,d]
                qz_enc_m - encoder density means  [N*T,2*q]
                qz_enc_logv - encoder density variances [N*T,2*q]
        '''
        [N,T,nc,d,d] = X.shape
        q = qz_m.shape[1]//2
        # prior
        log_pzt = self.mvn.log_prob(zode_L.contiguous().view([L*N*T,2*q])) # L*N*T
        log_pzt = log_pzt.view([L,N,T]) # L,N,T
        kl_zt = logpL - log_pzt  # L,N,T
        kl_z  = kl_zt.sum(2).mean(0) # N
        # likelihood
        XL = X.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        lhood_L = torch.log(XrecL)*XL + torch.log(1-XrecL)*(1-XL) # L,N,T,nc,d,d
        lhood = lhood_L.sum([2,3,4,5]).mean(0) # N
        if qz_enc_m is not None: # instant encoding
            qz_enc_mL    = qz_enc_m.repeat([L,1])  # L*N*T,2*q
            qz_enc_logvL = qz_enc_logv.repeat([L,1])  # L*N*T,2*q
            mean_ = qz_enc_mL.contiguous().view(-1) # L*N*T*2*q
            std_  = qz_enc_logvL.exp().contiguous().view(-1) # L*N*T*2*q
            qenc_zt_ode = Normal(mean_,std_).log_prob(zode_L.contiguous().view(-1)).view([L,N,T,2*q])
            qenc_zt_ode = qenc_zt_ode.sum([3]) # L,N,T
            inst_enc_KL = logpL - qenc_zt_ode
            inst_enc_KL = inst_enc_KL.sum(2).mean(0) # N
            return lhood.mean(),kl_z.mean(),inst_enc_KL.mean()
        else:
            return lhood.mean(),kl_z.mean() # mean over training samples

    def forward(self, X, L=1, inst_enc=False, method='dopri5', dt=0.1):
        ''' Returns
                Xrec_mu    - reconstructions from the mean embedding - [N,nc,D,D]
                Xrec_L     - reconstructions from latent samples     - [L,N,nc,D,D]
                qz_m       - mean of the latent embeddings           - [N,q]
                qz_logv    - log variance of the latent embeddings   - [N,q]
                lhood-kl_z - ELBO   
                lhood      - reconstruction likelihood
                kl_z       - KL
        '''
        # encode
        [N,T,nc,d,d] = X.shape
        h = self.encoder(X[:,0])
        qz0_m, qz0_logv = self.fc1(h), self.fc2(h) # N,2q & N,2q
        q = qz0_m.shape[1]//2
        # latent samples
        eps   = torch.randn_like(qz0_m)  # N,2q
        z0    = qz0_m + eps*torch.exp(qz0_logv) # N,2q
        logp0 = self.mvn.log_prob(eps) # N 
        # ODE
        t  = dt * torch.arange(T,dtype=torch.float).to(z0.device)
        ztL   = []
        logpL = []
        # sample L trajectories
        for l in range(L):
            self.mlp.sample_weights()
            zt,logp = odeint(self.odefp,(z0,logp0),t,method=method) # T,N,2q & T,N
            ztL.append(zt.permute([1,0,2]).unsqueeze(0)) # 1,N,T,2q
            logpL.append(logp.permute([1,0]).unsqueeze(0)) # 1,N,T
        ztL   = torch.cat(ztL,0) # L,N,T,2q
        logpL = torch.cat(logpL) # L,N,T
        # decode
        st_muL = ztL[:,:,:,q:] # L,N,T,q
        s = self.fc3(st_muL.contiguous().view([L*N*T,q]) ) # L*N*T,h_dim
        Xrec = self.decoder(s) # L*N*T,nc,d,d
        Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d
        # likelihood and elbo
        if inst_enc:
            h = self.encoder(X.contiguous().view([N*T,nc,d,d]))
            qz_enc_m, qz_enc_logv = self.fc1(h), self.fc2(h) # N*T,2q & N*T,2q
            lhood, kl_z, inst_KL = self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec, L, qz_enc_m, qz_enc_logv)
            elbo = lhood - kl_z - inst_KL
        else:
            lhood, kl_z = self.elbo(qz0_m, qz0_logv, ztL, logpL, X, Xrec, L)
            elbo = lhood - kl_z
        return Xrec, qz0_m, qz0_logv, ztL, elbo, lhood, kl_z

    def mean_rec(self, X, method='dopri5', dt=0.1):
        [N,T,nc,d,d] = X.shape
        # encode
        h = self.encoder(X[:,0])
        qz0_m = self.fc1(h) # N,2q
        q = qz0_m.shape[1]//2
        # ode
        def odefp(t,vs):
            q = vs.shape[1]//2
            dv = self.f(t,vs) # N,q 
            ds = vs[:,:q]  # N,q
            return torch.cat([dv,ds],1) # N,2q
        self.mlp.set_mean_weights()
        t  = dt * torch.arange(T,dtype=torch.float).to(qz0_m.device)
        zt_mu = odeint(odefp,qz0_m,t,method=method).permute([1,0,2]) # N,T,2q
        # decode
        st_mu = zt_mu[:,:,q:] # N,T,q
        s = self.fc3(st_mu.contiguous().view([N*T,q]) ) # N*T,q
        Xrec_mu = self.decoder(s) # N*T,nc,d,d
        Xrec_mu = Xrec_mu.view([N,T,nc,d,d]) # N,T,nc,d,d
        # error
        mse = torch.mean((Xrec_mu-X)**2)
        return Xrec_mu,mse
        
# plotting
def plot_rot_mnist(X,Xrec,show=False,fname='rot_mnist_inst.png'):
    N = min(X.shape[0],10)
    Xnp = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    T = X.shape[1]
    plt.figure(2,(T,3*N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+1)
            plt.imshow(np.reshape(Xnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+T+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.savefig(fname)
    if show is False:
        plt.close()


if __name__ == '__main__':
    freeze_support()
    odevae = ODE2VAE(q=8).to(device)
    Nepoch = 100
    optimizer = torch.optim.Adam(odevae.parameters(),lr=1e-3)
    for ep in range(Nepoch):
        for local_batch in training_generator:
            minibatch = local_batch.to(device)
            elbo, lhood, kl_z = odevae(minibatch, L=25, inst_enc=True, method='dopri5')[4:]
            tr_loss = -lhood
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
        with torch.set_grad_enabled(False):
            for test_batch in test_generator:
                test_batch = test_batch.to(device)
                Xrec_mu, test_mse = odevae.mean_rec(test_batch, method='dopri5')
                plot_rot_mnist(test_batch,Xrec_mu,False)
                torch.save(odevae.state_dict(), 'ode2vae_mnist.torch')
                break
        print('Epoch:{:4d}/{:4d}, tr_elbo:{:.3f}, test_mse:{:.3f}'.format(ep,Nepoch,tr_loss.item(),test_mse.item()))
