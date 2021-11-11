import torch
from torch import nn
class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stab_loss = StabLoss()
        self.shape_loss = ShapeLoss()
        self.temp_loss = TempLoss()

    def forward(self, f1, i1, f2, i2, g1, igt):
        stab = self.stab_loss(f1, i1, igt)
        shape = self.shape_loss(f1, g1)
        temp = self.temp_loss(f1, f2, i1, i2)

        return stab + shape + temp

class StabLoss(nn.Module):
    def __init__(self, alpha1=50.0, alpha2=1.0) -> None:
        super().__init__()
        self.pixel_loss = PixelLoss()
        self.feature_loss = FeatureLoss()

        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, f1, i1, igt):
        pixel = self.pixel_loss(f1, i1, igt)
        feature = self.feature_loss(f1, i1)

        return (self.alpha1 * pixel) + (self.alpha2 * feature)


class ShapeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class TempLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class PixelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stn_warpper = STN()
        self.MSE = nn.MSELoss()

    def forward(self, f1, i1, igt):
        new_i = self.stn_warpper(i1, f1)
        return self.MSE(new_i, igt)

class FeatureLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # load matches
    
    def forward(self, f1, i1):
        print('Implement Feature Alignment Loss!!')
        # use machts to compute feature alignment


        return None

class STN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, i, f):
        print("Do not forget to implement the STN")
        return i