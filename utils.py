import torch
from torch.functional import Tensor
import torchvision
from enum import Enum
import numpy as np
import scipy
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

"""
Enum to give the different types of datasets available
"""
class datasets(Enum):
    MNIST = 1
    SVHN = 2

"""
Loads the Mnist training data
returns a trainloader, validation loader and testloader
"""
def loadDataset(batch_size,dataset):
    #load data if mnist
    if dataset == dataset.MNIST:
        # transform to match svhn baseline
        baseline =torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.Grayscale(num_output_channels =3),
            torchvision.transforms.ToTensor()
        ])
        train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=baseline )

        train, validation = torch.utils.data.random_split(train, [50000, 10000])
        test = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=baseline)
    #load data if svhn
    elif dataset == dataset.SVHN:
        train = torchvision.datasets.SVHN(
            root='./data', split='train', download=True, transform=torchvision.transforms.ToTensor())
        validation = torchvision.datasets.SVHN(
            root='./data', split='extra', download=True, transform=torchvision.transforms.ToTensor())
        test = torchvision.datasets.SVHN(
            root='./data', split='test', download=True, transform=torchvision.transforms.ToTensor())
    else:
        raise Exception("Dataset not found, please use mnist or svhn") 

    #initialize loaders
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2)

    validationloader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, shuffle=True, num_workers=2)
            
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=2)

    return (trainloader, validationloader, testloader)


def train(network, data, epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt = torch.optim.Adam(network.parameters())
    for epoch in range(epochs):
        for i, (x, y) in enumerate(data):
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat,z, mu, sigma = network(x)
            loss = network.loss(x,x_hat,mu,sigma)
            loss.backward()
            opt.step()
            if i % 100 == 0:
                print(f'Epoch: [{epoch}/{epochs}], batch: {i}, Loss: {loss.item()}')
            if i % 3000 == 0:
                plt.rcParams['figure.figsize'] = [16, 5]
                print(y)
                fig, axs = plt.subplots(2, 8)
                for j in range(8):
                    axs[0,j].imshow(x[j].permute(1, 2, 0))
                    axs[1,j].imshow(x_hat[j].detach().permute(1, 2, 0))

                plt.subplots_adjust(left=0.1,
                    wspace=0.3, 
                    hspace=0.2)
                plt.show()



