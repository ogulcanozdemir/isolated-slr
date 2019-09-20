from utils.logger import logger

import torch.nn as nn
import torch


class MultiLayerLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 layer_type,
                 num_hidden=256,
                 num_layers=2,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False,
                 device=None):
        super(MultiLayerLSTM, self).__init__()

        logger.info('Initializing MultiLayerLSTM model ...')
        logger.info('Layer Type: {}'.format(layer_type.__name__))
        logger.info('Bidirectional: {}'.format(bidirectional))
        logger.info('Number of hidden units: {}'.format(num_hidden))
        logger.info('Number of layers: {}'.format(num_layers))
        logger.info('Dropout: {}'.format(dropout))

        self.layer_type = layer_type
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.device = device

        self.seq_layer = self.layer_type(input_size=self.input_size,
                                         hidden_size=self.num_hidden,
                                         batch_first=batch_first,
                                         dropout=dropout,
                                         num_layers=self.num_layers,
                                         bidirectional=self.bidirectional)

        self.num_directions = 2 if bidirectional else 1
        self.fc = None

    def forward(self, x):
        # lstm_out, self.hidden = self.seq_layer(x.view(len(x), self.batch_size, -1))
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.seq_layer(x, hidden)

        # if self.bidirectional:
        #     forward_output, backward_output = output[:-2, :, :self.num_hidden], output[2:, :, self.num_hidden:]

        fc_out = self.fc(output[:, -1, :])
        fc_out = fc_out.view(batch_size, -1)
        # lstm_out, self.hidden = self.seq_layer(x.permute(1,0,2))
        # out = self.fc(lstm_out[-1].view(self.batch_size, -1))
        return fc_out

    def init_hidden(self, batch_size):
        if self.layer_type == nn.LSTM:
            return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.num_hidden).to(self.device),
                    torch.zeros(self.num_layers * self.num_directions, batch_size, self.num_hidden).to(self.device))
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.num_hidden).to(self.device)
