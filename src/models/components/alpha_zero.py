import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# BLOCKS = 10
# GOBAN_SIZE = 9
# HISTORY = 7
# INPLANES = (HISTORY + 1) * 2 + 1
# OUTPLANES_MAP = 10
# OUTPLANES = (GOBAN_SIZE**2) + 1
# ALPHA = 0.03


def _prepare_state(state):
    """Transform the numpy state into a PyTorch tensor with cuda
    if available
    """
    x = torch.from_numpy(np.array([state]))
    x = torch.tensor(x, dtype=torch.float)
    return x


def get_version(folder_path, version):
    """Either get the last versionration of
    the specific folder or verify it version exists
    """
    if int(version) == -1:
        files = os.listdir(folder_path)
        if len(files) > 0:
            all_version = list(map(lambda x: int(x.split("-")[0]), files))
            all_version.sort()
            file_version = all_version[-1]
        else:
            return False
    else:
        test_file = f"{version}-extractor.pth.tar"
        if not os.path.isfile(os.path.join(folder_path, test_file)):
            return False
        file_version = version
    return file_version


def formate_state(state, probas, winner):
    """Repeat the probas and the winner to make every example identical after
    the dihedral rotation has been applied
    """
    probas = np.reshape(probas, (1, probas.shape[0]))
    probas = np.repeat(probas, 8, axis=0)
    winner = np.full((8, 1), winner)
    return state, probas, winner


class BasicBlock(nn.Module):
    """Basic residual block with 2 convolutions and a skip connection before the last
    ReLU activation.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class PolicyNet(nn.Module):
    """This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes, outplanes):
        super().__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(outplanes - 1, outplanes)

    def forward(self, x):
        """X : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = self.fc(x)
        probas = self.logsoftmax(x).exp()

        return probas


class ValueNet(nn.Module):
    """This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1.
    """

    def __init__(self, inplanes, outplanes):
        super().__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(outplanes - 1, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """X : feature maps extracted from the state
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = F.relu(self.fc1(x))
        winning = F.tanh(self.fc2(x))
        return winning


class Extractor(nn.Module):
    """This network is used as a feature extractor, takes as input the 'state' defined in
    the AlphaGo Zero paper
    - The state of the past n turns of the board (7 in the paper) for each player.
      This means that the first n matrices of the input state will be 1 and 0, where 1
      is a stone.
      This is done to take into consideration Go rules (repetitions are forbidden) and
      give a sense of time

    - The color of the stone that is next to play. This could have been a single bit, but
      for implementation purposes, it is actually expended to the whole matrix size.
      If it is black turn, then the last matrix of the input state will be a NxN matrix
      full of 1, where N is the size of the board, 19 in the case of AlphaGo.
      This is done to take into consideration the komi.

    The ouput is a series of feature maps that retains the meaningful informations
    contained in the input state in order to make a good prediction on both which is more
    likely to win the game from the current state, and also which move is the best one to
    make.
    """

    def __init__(self, inplanes, outplanes, blocks):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.blocks = blocks

        for block in range(blocks):
            setattr(self, f"res{block}", BasicBlock(outplanes, outplanes))

    def forward(self, x):
        """X : tensor representing the state
        feature_maps : result of the residual layers forward pass
        """
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(self.blocks - 1):
            x = getattr(self, f"res{block}")(x)

        feature_maps = getattr(self, f"res{self.blocks - 1}")(x)
        return feature_maps


# def load_player(folder, version):
#     """Load a player given a folder and a version"""
#     path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "saved_models")
#     if folder == -1:
#         folders = os.listdir(path)
#         folders.sort()
#         if len(folders) > 0:
#             folder = folders[-1]
#         else:
#             return False, False
#     elif not os.path.isdir(os.path.join(path, str(folder))):
#         return False, False

#     folder_path = os.path.join(path, str(folder))
#     last_version = get_version(folder_path, version)
#     if not last_version:
#         return False, False

#     return get_player(folder, int(last_version))


# def get_player(current_time, version):
#     """Load the models of a specific player"""
#     path = os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), "..", "saved_models", str(current_time)
#     )
#     try:
#         mod = os.listdir(path)
#         models = list(filter(lambda model: (model.split("-")[0] == str(version)), mod))
#         models.sort()
#         if len(models) == 0:
#             return False, version
#     except FileNotFoundError:
#         return False, version

#     player = Player()
#     checkpoint = player.load_models(path, models)
#     return player, checkpoint


# class Player:
#     def __init__(self):
#         """Create an agent and initialize the networks"""
#         self.extractor = Extractor(INPLANES, OUTPLANES_MAP)
#         self.value_net = ValueNet(OUTPLANES_MAP, OUTPLANES)
#         self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES)
#         self.passed = False

#     def predict(self, state):
#         """Predict the probabilities and the winner from a given state"""
#         feature_maps = self.extractor(state)
#         winner = self.value_net(feature_maps)
#         probas = self.policy_net(feature_maps)
#         return winner, probas

#     def save_models(self, state, current_time):
#         """Save the models"""
#         for model in ["extractor", "policy_net", "value_net"]:
#             self._save_checkpoint(getattr(self, model), model, state, current_time)

#     def _save_checkpoint(self, model, filename, state, current_time):
#         """Save a checkpoint of the models"""
#         dir_path = os.path.join(
#             os.path.dirname(os.path.realpath(__file__)), "..", "saved_models", current_time
#         )
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)

#         filename = os.path.join(dir_path, "{}-{}.pth.tar".format(state["version"], filename))
#         state["model"] = model.state_dict()
#         torch.save(state, filename)

#     def load_models(self, path, models):
#         """Load an already saved model"""
#         names = ["extractor", "policy_net", "value_net"]
#         for i in range(0, len(models)):
#             checkpoint = torch.load(os.path.join(path, models[i]))
#             model = getattr(self, names[i])
#             model.load_state_dict(checkpoint["model"])
#             return checkpoint
