import torch


class BCELossCustom:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.BCELoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight
    
class MSELossCustom:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.MSELoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight
