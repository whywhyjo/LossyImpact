import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils

class ConvNet_basic_two_fc(nn.Module):
    def __init__(self, ch_size, k_size, s_size, p_size ,final_size = 100):
        super(ConvNet_basic_two_fc, self).__init__()  
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, ch_size, kernel_size=k_size, stride=s_size, padding=p_size),
            nn.BatchNorm2d(ch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=k_size, stride=s_size))
        self.layer2 = nn.Sequential(
            nn.Conv2d(ch_size, ch_size*2,  kernel_size=k_size, stride=s_size, padding=p_size),
            nn.BatchNorm2d(ch_size*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=k_size, stride=s_size))
        self.layer3 = nn.Sequential(
            nn.Conv2d(ch_size*2, ch_size*4,  kernel_size=k_size, stride=s_size, padding=p_size),
            nn.BatchNorm2d(ch_size*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=k_size, stride=s_size))
        self.fc = nn.Sequential(        
            nn.Linear(final_size, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        initialize_weights(self)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


    
def initialize_weights(net):
    torch.manual_seed(1)
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass