import torch
from torch import nn
from torch import functional as F

class SNLIModel(nn.Module):
    def __init__(self, encoder):
        super(SNLIModel, self).__init__()
        self.encoder = encoder
        self.FC = nn.Sequential(nn.Linear(4*encoder.layer_size, 512), nn.ReLU(), nn.Dropout(encoder.dropout), nn.Linear(512, 3), nn.Softmax())
        self.name = encoder.name + " SNLI"
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, premise, hypothesis):
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        ### check if batch or single example
        if len(u.shape) == 1:
            cat_dim = 0
        else:
            cat_dim = 1
        inp = torch.cat((u, v, torch.abs(u-v), u*v), dim = cat_dim)
        return self.FC(inp)