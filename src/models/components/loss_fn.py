import torch


class BCELoss:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.BCELoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight


class BCEWithLogitsLoss:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight


class MSELoss:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.MSELoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight


class CrossEntropyLoss:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight


class HuberLoss:
    def __init__(self, tag, weight, delta=1.0):
        self.tag = tag
        self.weight = weight
        self.delta = delta
        self.criterion = torch.nn.SmoothL1Loss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight


class AlphaLoss(torch.nn.Module):
    """Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        super().__init__()

    def forward(self, winner, self_play_winner, probas, self_play_probas):
        value_error = (self_play_winner - winner) ** 2
        policy_error = torch.sum((-self_play_probas * (1e-6 + probas).log()), 1)
        total_error = (value_error.view(-1) + policy_error).mean()
        return total_error * self.weight
