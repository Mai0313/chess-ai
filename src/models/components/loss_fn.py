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


class CrossEntropyLossCustom:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight


class HuberLossCustom:
    def __init__(self, tag, weight, delta=1.0):
        self.tag = tag
        self.weight = weight
        self.delta = delta
        self.criterion = torch.nn.SmoothL1Loss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight
