from torchvision import models
import torch.nn as nn

# For CIFAR-10 dataset
class Densenet121forCifar10(nn.Module):
    def __init__(self, pretrained=True):
        super(Densenet121forCifar10, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 10)
        
    def forward(self, x):
        return self.model(x)
    
# For CIFAR-100 dataset
class Densenet121forCifar100(nn.Module):
    def __init__(self, pretrained=True):
        super(Densenet121forCifar100, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 100)
        
    def forward(self, x):
        return self.model(x)
