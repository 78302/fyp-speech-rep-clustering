import torch
import torch.nn as nn
import numpy as np
# import torchvision

# INPUT_SIZE = 40
# HIDDEN_SIZE = 512  # units inside the lstm
# # DROP_RATE = 0.2  # drop-out rate
# LAYERS = 4  # number of lstm layers

class toy_lstm(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, LAYERS):
        super(toy_lstm, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            # dropout=DROP_RATE,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 40)  # fully connected layer
        self.h_s = None
        self.h_c = None

    def forward(self, x):
        r_out, (h_s, h_c) = self.rnn(x)
        output = self.fc(r_out)
        return output