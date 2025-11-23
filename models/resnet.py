from torchvision import models
import torch.nn as nn

# Make the model compatible with TRADES (disable all inplace ReLUs)
def fix_inplace_relus(model):
    """
    Let all nn.ReLU(inplace=True) be changed to inplace=False.
    Avoid the computation graph being damaged during TRADES/PGD backpropagation.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model

# For CIFAR-10 dataset
class Resnet18forCifar10(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18forCifar10, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
    def forward(self, x):
        return self.model(x)
    
# For CIFAR-100 dataset
class Resnet18forCifar100(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18forCifar100, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        
    def forward(self, x):
        return self.model(x)
    
# For ImageNet-100 dataset
class Resnet18forImagenet100(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet18forImagenet100, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        
    def forward(self, x):
        return self.model(x)
    
# For Places365 dataset
class Resnet18forPlaces365(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18forPlaces365, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 365)
        
    def forward(self, x):
        return self.model(x)
    
# For CIFAR-10 dataset (Supplementary)
class Resnet50forCifar10(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50forCifar10, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
    def forward(self, x):
        return self.model(x)

