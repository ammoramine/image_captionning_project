from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models
from torchsummary import summary
from torch import nn
import torch

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

if __name__ == '__main__':

    device = torch.cuda.device("cuda")
    encoder = EncoderCNN(256).to(device)
    print(summary(encoder, torch.zeros(32, 3, 224, 224).to(device)))