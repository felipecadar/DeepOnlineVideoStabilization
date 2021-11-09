import torch
from torch import nn
from torch._C import _from_dlpack
import torchvision.models as models
from torchvision.models import resnet


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[1:-1]
        self.resnet50 = nn.Sequential(*modules)
        for p in self.resnet50.parameters():
            p.requires_grad = False

        self.conv1 =  nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        return self.resnet50(self.conv1(x))
        
class Regressor(nn.Module):
    def __init__(self, h=4, w=4):
        super().__init__()
        final_shape = (h + 1) * (w + 1) * 2
        fc2048 = nn.Linear(2048, 2048)
        fc1024 = nn.Linear(2048, 1024)
        fc512  = nn.Linear(1024, 512)
        fc_reg = nn.Linear(512, final_shape)
        
        self.sequence = nn.Sequential(fc2048, fc1024, fc512, fc_reg)


    def forward(self, x):
        out = self.sequence(x)
        return out

class StableNet(nn.Module):
    def __init__(self, h=4, w=4):
        super().__init__()
        self.encoder = Encoder()
        self.regressor = Regressor(h=h, w=w)

    def forward(self, i1, i2):
        enc1 = self.encoder(i1)
        enc2 = self.encoder(i2)

        #check batch sizes
        assert(enc1.shape[0] == enc2.shape[0])
        batch_size = enc1.shape[0]

        enc1 = enc1.view(batch_size, -1)
        enc2 = enc2.view(batch_size, -1)

        ft1 = self.regressor(enc1)
        ft2 = self.regressor(enc2)

        return ft1, ft2        

if __name__ == "__main__":

    W = 512
    H = 288
    c = 7
    batch_size = 1

    x1 = torch.rand([batch_size, c, W, H])
    x2 = torch.rand([batch_size, c, W, H])

    model = StableNet()
    out1, out2 = model(x1, x2)

    print(out1.shape, out2.shape)