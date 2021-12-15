import torch
import torch.nn as nn 

  
class VAEgaus(nn.Module): 
    def __init__(self, D, M): 
        super(VAEgaus, self).__init__() 
        self.D = D 
        self.M = M 
  
        self.enc1 = nn.Linear(self.D, 400) 
        self.enc2 = nn.Linear(400, self.M*2) 
        self.dec1 = nn.Linear(self.M, 400) 
        self.dec2 = nn.Linear(400, self.D)

    def encode(self,x):
        batchSize = x.size()[0]
        x = torch.flatten(x,1)
        x = nn.functional.relu(self.enc1(x)) 
        x = self.enc2(x).view(-1, 2, self.M) 
        # get mean and log-std 
        mu = x[:, 0, :] 
        sigma = torch.exp(x[:, 1, :])
        # reparameterization 
        z = self.reparameterize(mu, sigma) 
        return (batchSize, z, mu, sigma)

    def decode(self,z,batchSize):
        x_hat = nn.functional.relu(self.dec1(z)) 
        x_hat = self.dec2(x_hat) 
        x_hat = x_hat.reshape((batchSize,3,32,32))
        return(x_hat)

    def reparameterize(self, mu, sigma): 
        eps = torch.randn_like(sigma) 
        z = mu + (eps * sigma) 
        return z

    def forward(self, x): 
        batchSize, z, mu, sigma = self.encode(x)
        x_hat = self.decode(z,batchSize)
        return (x_hat,z, mu, sigma)

    def loss(self, x, x_hat, mu, sigma): 
        RE = torch.nn.functional.mse_loss(x, x_hat,reduction='mean') 
        KL = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return RE + KL

