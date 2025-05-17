import torch
from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim


        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),   
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),        
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),        
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
            nn.Sigmoid()  
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
