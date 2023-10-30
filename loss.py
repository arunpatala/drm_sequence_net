
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, factor=2.0):
        super().__init__()
        weights = [factor, 1.0, 1.0/factor]
        self.weights = torch.Tensor(weights*4+[1.0]).unsqueeze(0)  # The weights should be defined once, not twice
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Use 'none' to avoid automatic mean calculation

    def forward(self, srcY, tgtY):
        weights = self.weights.to(srcY.device)
        losses = self.criterion(srcY.view(-1, srcY.shape[-1]), tgtY.view(-1)).view_as(tgtY)
        #print("losses", losses.shape, "weights", weights.shape)
        weighted_losses = losses * weights
        return weighted_losses.mean()
