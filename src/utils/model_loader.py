import os
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
import rootutils

import hydra

rootutils.setup_root(os.path.abspath("."), indicator=".project-root", pythonpath=True)


def get_correct_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("net.", "")  # remove "net." from the keys
        new_state_dict[name] = v
    return new_state_dict


def get_tempfix_for_torch(ckpt):
    """TODO(mai0313): remove _orig_mod. from the state_dict due to pytorch issue #101107.

    Ref: https://discuss.pytorch.org/t/how-to-save-load-a-model-with-torch-compile/179739/2
         https://github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089
    In short, when you train a model with torch.compile, it will add _orig_mod. to the state_dict, which is not what we need;
    So we just simply remove it.
    """
    new_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        name = k.replace("_orig_mod.", "")
        new_dict[name] = v
    return new_dict


def load_model_from_path(log_directory):
    ckpt_path = f"{log_directory}/checkpoints/last.ckpt"
    model_config = OmegaConf.load(f"{log_directory}/.hydra/config.yaml")
    compile_option = model_config.model.compile
    if compile_option:
        model_instance = hydra.utils.instantiate(model_config.model)
        checkpoint = torch.load(ckpt_path)
        fixed_state_dict = get_tempfix_for_torch(checkpoint)
        model_instance.load_state_dict(fixed_state_dict)
    else:
        model_instance = hydra.utils.instantiate(model_config.model)
        model_instance.load_from_checkpoint(ckpt_path)
    return model_instance
