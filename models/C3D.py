from constants import WEIGHT_INIT
import torch
import torch.nn as nn


class C3D(nn.Module):

    def __init__(self, num_classes, batch_norm=True, dropout_prob=0, weight_initializer=None):
        super(C3D, self).__init__()

        self.batch_norm = batch_norm
        self.dropout_prob = dropout_prob
        self.weight_initializer = weight_initializer

        print('Initializing C3D model ...')
        print('Batch normalization (after conv): {}'.format(batch_norm))
        print('Dropout [fc_6, fc_7]: {}'.format(dropout_prob))
        print('Weight initializer: {}'.format(weight_initializer))

        self.conv_1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv_2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv_3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv_4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv_5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv_5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool_5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc_6 = nn.Linear(8192, 4096)
        self.fc_7 = nn.Linear(4096, 4096)
        self.fc_8 = nn.Linear(4096, num_classes)

        if self.batch_norm is not None:
            self.conv_bn_1a = nn.BatchNorm3d(64)
            self.conv_bn_2a = nn.BatchNorm3d(128)
            self.conv_bn_3a = nn.BatchNorm3d(256)
            self.conv_bn_3b = nn.BatchNorm3d(256)
            self.conv_bn_4a = nn.BatchNorm3d(512)
            self.conv_bn_4b = nn.BatchNorm3d(512)
            self.conv_bn_5a = nn.BatchNorm3d(512)
            self.conv_bn_5b = nn.BatchNorm3d(512)

        if self.dropout_prob != 0:
            self.dropout = nn.Dropout(p=self.dropout_prob)

        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        # block 1
        x = self.conv_1a(x)
        if self.batch_norm:
            x = self.conv_bn_1a(x)
        x = self.relu(x)
        x = self.pool_1(x)

        # block 2
        x = self.conv_2a(x)
        if self.batch_norm:
            x = self.conv_bn_2a(x)
        x = self.relu(x)
        x = self.pool_2(x)

        # block 3a
        x = self.conv_3a(x)
        if self.batch_norm:
            x = self.conv_bn_3a(x)
        x = self.relu(x)

        # block 3b
        x = self.conv_3b(x)
        if self.batch_norm:
            x = self.conv_bn_3b(x)
        x = self.relu(x)
        x = self.pool_3(x)

        # block 4a
        x = self.conv_4a(x)
        if self.batch_norm:
            x = self.conv_bn_4a(x)
        x = self.relu(x)

        # block 4b
        x = self.conv_4b(x)
        if self.batch_norm:
            x = self.conv_bn_4b(x)
        x = self.relu(x)
        x = self.pool_4(x)

        # block 5a
        x = self.conv_5a(x)
        if self.batch_norm:
            x = self.conv_bn_5a(x)
        x = self.relu(x)

        # block 5b
        x = self.conv_5b(x)
        if self.batch_norm:
            x = self.conv_bn_5b(x)
        x = self.relu(x)
        x = self.pool_5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc_6(x))
        if self.dropout_prob != 0:
            x = self.dropout(x)
        x = self.relu(self.fc_7(x))
        if self.dropout_prob != 0:
            x = self.dropout(x)

        x = self.fc_8(x)

        return x

    def init_weights(self):
        print('Initializing weights using {} ...'.format(self.weight_initializer))

        weight_init_fun = WEIGHT_INIT[self.weight_initializer]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                weight_init_fun(m.weight)
                m.bias.data.fill_(1)
            elif isinstance(m, nn.Linear):
                weight_init_fun(m.weight)
                m.bias.data.fill_(1)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_weights(self, weights=None):
        if weights is None:
            raise FileNotFoundError

        layer_map = {
                     # Conv1
                     "features.0.weight": "conv1.weight",
                     "features.0.bias": "conv1.bias",
                     # Conv2
                     "features.3.weight": "conv2.weight",
                     "features.3.bias": "conv2.bias",
                     # Conv3a
                     "features.6.weight": "conv3a.weight",
                     "features.6.bias": "conv3a.bias",
                     # Conv3b
                     "features.8.weight": "conv3b.weight",
                     "features.8.bias": "conv3b.bias",
                     # Conv4a
                     "features.11.weight": "conv4a.weight",
                     "features.11.bias": "conv4a.bias",
                     # Conv4b
                     "features.13.weight": "conv4b.weight",
                     "features.13.bias": "conv4b.bias",
                     # Conv5a
                     "features.16.weight": "conv5a.weight",
                     "features.16.bias": "conv5a.bias",
                     # Conv5b
                     "features.18.weight": "conv5b.weight",
                     "features.18.bias": "conv5b.bias",
                     # fc6
                     "classifier.0.weight": "fc6.weight",
                     "classifier.0.bias": "fc6.bias",
                     # fc7
                     "classifier.3.weight": "fc7.weight",
                     "classifier.3.bias": "fc7.bias",
                     }

        p_dict = torch.load(weights)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in layer_map:
                continue
            s_dict[layer_map[name]] = p_dict[name]
        self.load_state_dict(s_dict, strict=False)

    # def get_1x_lr_params(self):
    #     b = [self.conv_1a, self.conv_2a, self.conv_3a, self.conv_3b, self.conv_4a, self.conv_4b,
    #          self.conv_5a, self.conv_5b, self.fc_6, self.fc_7]
    #     for i in range(len(b)):
    #         for k in b[i].parameters():
    #             if k.requires_grad:
    #                 yield k
    #
    # def get_10x_lr_params(self):
    #     b = [self.fc_8]
    #     for j in range(len(b)):
    #         for k in b[j].parameters():
    #             if k.requires_grad:
    #                 yield k
