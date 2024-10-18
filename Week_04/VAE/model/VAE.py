import torch.nn as nn
import torch
import torch.nn.functional as F
# VAE 정의
class VAE(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, latent_size=64):
        super(VAE, self).__init__()
        self.en_1 = nn.Linear(input_size, hidden_size)
        self.en_mu = nn.Linear(hidden_size, latent_size)
        self.en_logvar = nn.Linear(hidden_size, latent_size)
        self.de_1 = nn.Linear(latent_size, hidden_size)
        self.de_2 = nn.Linear(hidden_size, input_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        
    def encoder(self, x):
        h = self.leaky_relu(self.en_1(x))
        mu = self.en_mu(h)
        logvar = self.en_logvar(h)
        return mu, logvar
    
    def decoder(self, z):
        h = self.leaky_relu(self.de_1(z))
        x_hat = torch.sigmoid(self.de_2(h))
        return x_hat

    def reparameterize(self, mu, logvar):  # log var * 0.5 = log std because of log/
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z 
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar

# loss function 정의
def loss_function(x_hat, data, mu, logvar):
    data = data.view(data.size(0), -1)
    reconstruction_loss = F.binary_cross_entropy(x_hat, data, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # E_q[log q(z|x) - log p(z)] = E_q[-0.5*(logvar + (z-mu)^2/var - log(2pi) - z^2)] = -0.5*(logvar + 1 - mu^2 - var) 
    return reconstruction_loss + kl_divergence