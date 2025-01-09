from math import ceil
import torch
import torch.nn as nn
from utils import spectro
from einops import rearrange

EPSILON = 1.0e-7

class DrumClassifierModel(nn.Module):
    def __init__(self,cfg):
        super(DrumClassifierModel,self).__init__()
        self.sr = cfg.audio.sr
        self.tl = cfg.audio.max_l
        self.num_targets = len(cfg.train.targets)
        self.n_fft = cfg.model.nfft
        self.growth = cfg.model.growth
        self.conv_depth = cfg.model.conv_depth
        #conv2d stack
        self.in_c = 2
        self.gelu = nn.GELU()
        self.conv_stack = nn.ModuleList()
        for i in range(self.conv_depth):
            self.conv_stack.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.in_c,
                        self.in_c*self.growth,
                        kernel_size=[4,5],
                        stride=[2,1],
                        padding=[1,2]
                    ),
                    nn.BatchNorm2d(self.in_c*self.growth),
                    nn.Dropout(0.1),
                    nn.GELU()
                )
            )
            self.in_c *= self.growth
        self.do = nn.Dropout(0.2)
        final_f_channls = (self.n_fft//2) // (2**self.conv_depth)
        self.t_bins = ceil(cfg.audio.max_l * cfg.audio.sr / (self.n_fft//4)) #the 4 coming from
        #the fact the STFT uses a hop size of n_fft/4 by default in this case
        self.bidirectional = cfg.model.bidir
        self.trnn = nn.RNN(self.in_c*final_f_channls, final_f_channls,batch_first=True,bidirectional=self.bidirectional)#seq = t
        self.frnn = nn.RNN(self.t_bins, 64, batch_first=True,bidirectional=False)
        multi = 2 if self.bidirectional else 1
        self.do2 = nn.Dropout(0.3)
        self.fc = nn.Linear(final_f_channls*multi*64, final_f_channls*multi)
        self.fc2 = nn.Linear(final_f_channls*multi, final_f_channls)
        self.fc3 = nn.Linear(final_f_channls, self.num_targets)
        
        self.flatten = nn.Flatten()

    def forward(self,input):
        #x: [B,2,N] -> 4 seconds samples
        #out: [B,C] -> probability output for each class
        
        #normalize audio data
        meant = input.mean(dim=(1, 2), keepdim=True)
        stdt = input.std(dim=(1, 2), keepdim=True)
        input = (input-meant)/(stdt+EPSILON)
        #go to the freq domain
        x = spectro(input, n_fft=self.n_fft)
        x = abs(x).to(input.device)
        x = x[:,:,:-1,:]
        for i in range(self.conv_depth):
            x = self.conv_stack[i](x)
        #B,C,F,T
        x = rearrange(x, 'b c f t -> b t (c f)')
        x,_ = self.trnn(x)
        x = self.do(x)
        x = self.gelu(x)
        x = rearrange(x, 'b t f-> b f t')
        x,_ = self.frnn(x)
        x = self.do(x)
        x = self.gelu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.do2(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.do2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        return x