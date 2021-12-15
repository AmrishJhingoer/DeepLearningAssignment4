from utils import *
from vaeGaus import VAEgaus

if __name__ == '__main__':
    
    trainloader, validtaionloader, testloader = loadDataset(16,datasets.MNIST)

    D = 32*32*3
    M = 10
    nn = VAEgaus(D,M)
    print("--Network initialized")
    train(nn,trainloader,5)
