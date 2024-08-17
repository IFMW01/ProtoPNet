import torch.optim as optim
import torch
import torch.nn as nn
import os
import numpy as np
import random



# Sets the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Counts the number of parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Sets model hyperparameters
def set_hyperparameters(model,architecture,lr):
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    return optimizer,criterion

# Gets the device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Creates directory
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)













     