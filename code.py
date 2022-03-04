
# In net2, we encode content and motion indiviually and then combine them.
import torch
import torch.nn as nn

from convlstm import ConvLSTM 

# from torchkeras import Model,summary

# from torchsummary import summary

import torch.utils.data as Data

##### change point!
# import data1 as datt
import numpy as np

# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



t1 = 20
t2 = 20


class MetNet(nn.Module):
   
    def __init__(self,ngpu=2):
        super(MetNet, self).__init__()
        self.ngpu = ngpu
       # (b,c,t,h,w)
        self.conv1 = nn.Sequential(
        nn.Conv3d(in_channels = 4,out_channels = 64,kernel_size=(3,3,3),stride=1,padding =(1,1,1)),
        nn.LeakyReLU(),
        
        nn.Conv3d(in_channels = 64,out_channels = 128,kernel_size = (3,3,3),stride=1,padding=(1,1,1)),
        nn.LeakyReLU(),
        nn.Conv3d(in_channels = 128,out_channels = 256,kernel_size = (3,3,3),stride=1,padding=(1,1,1)),
        nn.LeakyReLU()
        )

        self.lstm1 = ConvLSTM(4, 32, (3,3), 2, True, True, False)
        

        self.lstm2 = ConvLSTM(32, 64, (3,3), 2, True, True, False) 



        self.conv2 = nn.Sequential(
        nn.Conv3d(in_channels = 320,out_channels = 180,kernel_size = (3,3,3),stride=1,padding=(1,1,1)),
        nn.LeakyReLU(),    
        nn.ConvTranspose3d(in_channels = 180,out_channels = 32,kernel_size = (3,3,3),stride=1,padding=(1,1,1)),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(in_channels = 32,out_channels = 4,kernel_size = (3,3,3),stride=1,padding=(1,1,1)),
        nn.LeakyReLU(),
        )
        
    
    def forward(self,input):
        output1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))

        input2 = input.contiguous().view(-1, 20, 4, 96, 96)
        output2 = nn.parallel.data_parallel(self.lstm1,input2 , range(self.ngpu))
       
        output2 = output2[0][0]
        output3 = nn.parallel.data_parallel(self.lstm2,output2 , range(self.ngpu))
        output3 = output3[0][0].permute(0, 2, 1, 3, 4)

        inn = torch.cat((output1,output3),dim= 1)

        output = nn.parallel.data_parallel(self.conv2, inn, range(self.ngpu))

        return output









    