import torch.nn as nn 
import torch
  
class GAN(nn.Module): 
    def __init__(self, D, M): 
        super(GAN, self).__init__() 
        self.D = D 
        self.M = M 
  
        self.gen1 = nn.Linear(in_features= self.M, out_features=300) 
        self.gen2 = nn.Linear(in_features=300, out_features= self.D) 
        self.dis1 = nn.Linear(in_features= self.D, out_features=300) 
        self.dis2 = nn.Linear(in_features=300, out_features=1)

    def generate(self, N): 
            z = torch.randn(size=(N, self.D)) 
            x_gen = self.gen1(z) 
            x_gen = nn.functional.relu(x_gen) 
            x_gen = self.gen2(x_gen) 
            return x_gen 
    def discriminate(self, x): 
        y = self.dis1(x) 
        y = nn.functional.relu(y) 
        y = self.dis2(y) 
        y = torch.sigmoid(y) 
        return y
        
    def gen_loss(self, d_gen): 
       return torch.log(1. - d_gen) 
    def dis_loss(self, d_real, d_gen): 
       # We maximize wrt. the discriminator, but optimizers minimize! 
       # We need to include the negative sign! 
       return -(torch.log(d_real) + torch.log(1. - d_gen)) 
    def forward(self, x_real): 
       x_gen = self.generate(N=x_real.shape[0]) 
       d_real = self.discriminate(x_real) 
       d_gen = self.discriminate(x_gen) 
       return d_real, d_gen