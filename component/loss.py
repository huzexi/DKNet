from torch import nn


class Loss(nn.Module):
    def __init__(self, l_config):
        super().__init__()
        self.l_config = l_config
        self.mse_loss = nn.MSELoss()
        # self.tv_loss = TVLoss()

    def forward(self, pred, gt):
        loss = self.mse_loss(pred, gt)
        return loss
