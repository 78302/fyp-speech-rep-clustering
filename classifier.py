import torch
import torch.nn as nn
import numpy as np

from functions import load_model
from apc import toy_lstm
from vqapc import toy_vqapc


class classification_net(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, PRETRAIN_PATH=None):
        super(classification_net, self).__init__()

        self.PRE_TRAIN = False
        self.OUTPUT_SIZE = OUTPUT_SIZE

        if PRETRAIN_PATH:  # use a pretrain model
            rnn_pretrain_model = toy_lstm(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4) # toy_lstm
            optimizer_pretrain = torch.optim.Adam(rnn_pretrain_model.parameters(), lr=0.001)  # just attach but no use
            rnn_pretrain_model, optimizer_pretrain = load_model(PRETRAIN_PATH, rnn_pretrain_model, optimizer_pretrain) # load the model
            self.pre_train_model = nn.Sequential(*list(rnn_pretrain_model.children())[:-1])  # only take the LSTM part

            for p in self.parameters():  # freeze the pretrained mode parameters
                p.requires_grad = False

            self.PRE_TRAIN = True

            # rnn_pretrain.eval()
        # self.hidden_layer = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.out = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        if self.PRE_TRAIN:
            x = torch.unsqueeze(x, 0)
            x, states = self.pre_train_model(x)
            x = x[0]
        # x = torch.relu(self.hidden_layer(x))
        x = self.out(x)
        # x = torch.nn.functional.softmax(x)
        return x


if __name__ == '__main__':

    net1 = classification_net(INPUT_SIZE=40, HIDDEN_SIZE=512, OUTPUT_SIZE=43)
    # print(net1)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #

    net2 = classification_net(INPUT_SIZE=512, HIDDEN_SIZE=512, OUTPUT_SIZE=43, PRETRAIN_PATH='./model/Epoch1.pth.tar').to(DEVICE)
    print(net2)

    for n in filter(lambda p: p.requires_grad, net2.parameters()):
        print(n.shape)

    # for name,parameters in net2.named_parameters():
    #     if parameters.requires_grad:
    #         print(name,'__has-grad__:',parameters.size())
    #     else:
    #         print(name,'__no-grad__:',parameters.size())
