# Model Implementation

import torch
import torch.nn as nn
from torch.types import FileLike


#TODO: it might be necessary do define Module subclasses to elegantly define the model (like in Homework 2)
class AIAYNModel(nn.Module):
    def __init__(self) -> None:
        super(AIAYNModel, self).__init__()
        #TODO define layers

        #TODO remove placeholder linear impl
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        #TODO define forward pass
        #TODO remove placeholder linear impl
        return self.linear(x)

    def save(self, path: FileLike) -> None:
        torch.save(self.state_dict(), path)
        pass

    def load(self, path: FileLike) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))

