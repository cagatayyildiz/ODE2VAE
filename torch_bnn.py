import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Normal, Bernoulli, kl_divergence as kl


def get_act(act="relu"):
    if act=="relu":         return nn.ReLU()
    elif act=="elu":        return nn.ELU()
    elif act=="celu":       return nn.CELU()
    elif act=="leaky_relu": return nn.LeakyReLU()
    elif act=="sigmoid":    return nn.Sigmoid()
    elif act=="tanh":       return nn.Tanh()
    elif act=="linear":     return nn.Identity()
    elif act=='softplus':   return nn.modules.activation.Softplus()
    else:                   return None


class BNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid_layers: int=2, act: str='softplus', dropout=0.0, \
                        n_hidden: int=100, requires_grad=True, logsig0=-3, bnn=True, layer_norm=False):
        super().__init__()
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        self.weight_mus  = nn.ParameterList([])
        self.bias_mus    = nn.ParameterList([])
        self.layer_norms = nn.ModuleList([])
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.acts    = []
        self.act = act 
        self.bnn = bnn
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weight_mus.append(Parameter(torch.Tensor(n_out, n_in),requires_grad=requires_grad))
            self.bias_mus.append(Parameter(torch.Tensor(n_out),requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
            self.layer_norms.append(nn.LayerNorm(n_out) if layer_norm and i<n_hid_layers else nn.Identity())
        if bnn:
            self.weight_logsigs = nn.ParameterList([])
            self.bias_logsigs   = nn.ParameterList([])
            self.logsig0 = logsig0
            for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
                self.weight_logsigs.append(Parameter(torch.Tensor(n_out, n_in),requires_grad=requires_grad))
                self.bias_logsigs.append(Parameter(torch.Tensor(n_out),requires_grad=requires_grad))
        self.reset_parameters()

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weight_mus,self.bias_mus)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
        for norm in self.layer_norms[:-1]:
            if isinstance(norm,nn.LayerNorm):
                norm.reset_parameters()
        if self.bnn:
            for w,b in zip(self.weight_logsigs,self.bias_logsigs):
                nn.init.uniform_(w,self.logsig0-1,self.logsig0+1)
                nn.init.uniform_(b,self.logsig0-1,self.logsig0+1)

    def __sample_weights(self,mean=False):
        if self.bnn and not mean:
            weights = [weight_mu + torch.randn_like(weight_mu)*weight_sig.exp() \
                for weight_mu,weight_sig in zip(self.weight_mus,self.weight_logsigs)]
            biases  = [bias_mu + torch.randn_like(bias_mu)*bias_sig.exp() \
                for bias_mu,bias_sig in zip(self.bias_mus,self.bias_logsigs)]
        else:
            weights = self.weight_mus
            biases  = self.bias_mus
        return weights,biases

    def draw_f(self,mean=False):
        weights,biases = self.__sample_weights(mean)
        def f(x):
            for (weight,bias,act,norm) in zip(weights,biases,self.acts,self.layer_norms):
                x = norm(act(self.dropout(F.linear(x,weight,bias))))
            return x
        return f

    def forward(self, x):
        return self.draw_f()(x)

    def kl(self):
        w_mus = [weight_mu.view([-1]) for weight_mu in self.weight_mus]
        b_mus = [bias_mu.view([-1]) for bias_mu in self.bias_mus]
        mus = torch.cat(w_mus+b_mus)
        w_logsigs = [weight_logsig.view([-1]) for weight_logsig in self.weight_logsigs]
        b_logsigs = [bias_logsigs.view([-1]) for bias_logsigs in self.bias_logsigs]
        sigs = torch.cat(w_logsigs+b_logsigs).exp()
        q = Normal(mus,sigs)
        N = Normal(torch.zeros(len(mus),device=mus.device),torch.ones(len(mus),device=mus.device))
        return kl(q,N)

    def __repr__(self):
        str_ = 'dropout rate = {:.2f}\n'.format(self.dropout_rate)
        for i,(weight,act) in enumerate(zip(self.weight_mus,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_
