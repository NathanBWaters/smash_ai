'''
Predicts which characters are fighting in the scene
'''
import os
import torch
import torch.nn as nn

from wavedash.constants import (
    CHARACTERS,
    NUM_PLAYERS,
    MODEL_CHECKPOINTS
)


class LeNetLayer(nn.Module):
    '''
    Simple LeNet layer
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 pool_dims=None):
        '''
        Create a single LeNet layer
        '''
        super(LeNetLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride)

        self.pool_dims = pool_dims

    def forward(self, x):
        '''
        Feed forward through one LeNet layer
        '''
        x = self.conv(x)
        x = torch.relu(x)

        if self.pool_dims:
            x = torch.max_pool2d(x, self.pool_dims)

        return x


class CharacterPredictModel(nn.Module):
    '''
    Net that learns to classify an Arabic letter from an image
    '''
    def __init__(self, load_weights=True):
        '''
        Creating the network
        '''
        super(CharacterPredictModel, self).__init__()
        # self.layer1 = LeNetLayer(3, 32, pool_dims=2)
        # self.layer2 = LeNetLayer(32, 32, pool_dims=2)
        # self.layer3 = LeNetLayer(32, 32, pool_dims=2)
        # self.layer4 = LeNetLayer(32, 32, pool_dims=2)
        # self.layer5 = LeNetLayer(32, 32)

        self.layer1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1)
        self.layer2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1)
        self.layer3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1)
        self.layer4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1)

        # this is our final classification layer
        self.fc1 = nn.Linear(832, len(CHARACTERS) * NUM_PLAYERS)

        if load_weights:
            self.load_weights()

    def forward(self, x):
        '''
        Forward prop
        '''
        x = self.layer1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.layer2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.layer3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.layer4(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        # x = self.layer5(x)

        # import pdb; pdb.set_trace()
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        output = torch.sigmoid(x)

        return output

    def load_weights(self):
        '''
        Loads the weights for the model
        '''
        print('Loading the weights for CharacterPredictModel')
        weights = torch.load(os.path.join(
            MODEL_CHECKPOINTS, '2_fixed_softmax_sigmoid_error__400'))
        self.load_state_dict(weights)

    def format_output(self, output):
        '''
        Formats the outupt of the model
        '''
        value, index = torch.max(
            torch.squeeze(output, 0), 0)
        confidence = torch.exp(value).item()
        class_id = index.item()
        return class_id, confidence

    def predict(self, image):
        '''
        Predict on an image.  Returns the class_id and the confidence
        '''
        self.eval()

        with torch.no_grad():
            output = self(torch.unsqueeze(image, 0))
            return self.format_output(output)
