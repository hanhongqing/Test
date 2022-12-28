import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvEncoder(nn.Module):
    def __init__(self, args):
        super(ConvEncoder,self).__init__()
        self.c = args.nn_n_channels
        self.a = args.size_patch // 2**4
        self.conv1 = nn.Conv2d(2,       self.c*1, 3, 1,1, bias = True)
        self.conv2 = nn.Conv2d(self.c*1,self.c*1, 3, 1,1, bias = True)
        self.conv3 = nn.Conv2d(self.c*1,self.c*1, 3, 1,1, bias = True)
        self.conv4 = nn.Conv2d(self.c*1,self.c*1, 3, 1,1, bias = True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.a*self.a*self.c*1,args.nn_dim_latent)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, args):
        super(ConvDecoder,self).__init__()
        self.c = args.nn_n_channels
        self.a = args.size_patch // 2**4
        self.t_conv1 = nn.ConvTranspose2d(self.c*1, self.c*1, 3, 1,1, bias = True)
        self.t_conv2 = nn.ConvTranspose2d(self.c*1, self.c*1, 3, 1,1, bias = True)
        self.t_conv3 = nn.ConvTranspose2d(self.c*1, self.c*1, 3, 1,1, bias = True)
        self.t_conv4 = nn.ConvTranspose2d(self.c*1, 2,        3, 1,1, bias = True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()
        self.fc = nn.Linear(args.nn_dim_latent,self.a*self.a*self.c*1)

    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.shape[0],-1,self.a,self.a)
        x = self.relu(self.t_conv1(self.upsample(x)))
        x = self.relu(self.t_conv2(self.upsample(x)))
        x = self.relu(self.t_conv3(self.upsample(x)))
        x = self.t_conv4(self.upsample(x))
        return x

class ConvAE(nn.Module):
    
    def __init__(self, args):
        super(ConvAE,self).__init__()
        
        self.encoder = ConvEncoder(args)
        self.decoder = ConvDecoder(args)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x





