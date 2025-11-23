import torch
import torch.nn as nn

class NormalizationWrapper(nn.Module):
    def __init__(self, model, dataset_name, device):
        super().__init__()
        self.model = model
        
        # Define mean and std based on the dataset
        if dataset_name == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif dataset_name == 'cifar100':
            mean = [0.5071, 0.4867, 0.4408]
            std = [0.2675, 0.2565, 0.2761]
        elif 'imagenet' in dataset_name:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else: # Default fallback
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1).to(device))

    def forward(self, x):
        # Normalize input x: [0, 1] -> Normalized [-2, 2]
        x = (x - self.mean) / self.std
        return self.model(x)

