import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import model.backbones as Backbones
from model.pooling import *
from utils.model_util import load_checkpoint


class CNNRNN(BaseModel):
    def __init__(self, backbone, pretrained_path=None, embedding_size=128, num_layers=1,
                 bidirectional=False, pooling='max', dropout=0, **kwargs):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.cnn = getattr(Backbones, backbone)(**kwargs)

        if pretrained_path is not None:
            state_dict = load_checkpoint(pretrained_path)
            self.cnn.load_state_dict(state_dict)
            # explicitly fix fc although it won't be used during training
            for param in self.cnn.fc.parameters():
                param.require_grad = False

        self.rnn = nn.LSTM(self.cnn.fc.in_features, embedding_size, num_layers=num_layers,
                           bidirectional=bidirectional, batch_first=True)

        if pooling == 'avg':
            self.pooling = AvgPooling()
        elif pooling == 'max':
            self.pooling = MaxPooling()
        else:
            raise NotImplementedError

        self.fc = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(True), nn.Dropout(dropout),
                                nn.Linear(embedding_size, 1))

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.view(B*T, C, H, W).contiguous()

        x = self.cnn(x, return_embedding=True) # (B*T, D)

        x = x.view(B, T, -1)
        self.rnn.flatten_parameters() # for strange warning if mute this line
        x = self.rnn(x)[0]
        if self.bidirectional:
            x = torch.mean(torch.stack(torch.chunk(x, 2, dim=-1)), dim=0)

        x = self.pooling(x) # (B, D)

        y = self.fc(x).squeeze(-1) # (B,)

        return y


# if __name__ == '__main__':
#     model = CNNRNN('EfficientFace')
#     x = torch.randn(4, 3, 8, 128, 128)
#     y = model(x)
#     print(y.size())
#     print(model)




