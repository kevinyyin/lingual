# Vendored unchanged from github.com/microsoft/ASL-citizen-code (MIT), ST-GCN/architecture/.py (originally AI4Bharat OpenHands, Apache-2.0).
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x