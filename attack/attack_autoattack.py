import sys
import logging
import os

# Add the project root directory to Python's module search path
# This works for both .py and .ipynb files
project_root = os.path.dirname(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Already added '{project_root}' to sys.path")

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import constants

from data.dataset import CustomImageDataset, AEDataset
from data.transform import train_transform, val_test_transform, train_transform_224, val_test_transform_224

from models.alicnn import AliCNN, AliCNNforCifar100, AliCNNforPlaces365, AliCNNforImagenet100
from models.cnn import CNNModel, CNNModelRL, CNNModelforCifar100, CNNModelforPlaces365, CNNModelforImagenet100
from models.densenet import Densenet121forCifar10, Densenet121forCifar100
from models.resnet import Resnet18forCifar10, Resnet50forCifar10, Resnet18forCifar100, Resnet18forPlaces365, Resnet18forImagenet100

import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import random
import json

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from sl_train.attack_autoattack import AutoAttack

def load_data(args):
    if args.dataset == 'cifar10':
        dataset_path = constants.CIFAR10_PATH
    elif args.dataset == 'cifar100':
        dataset_path = constants.CIFAR100_PATH
    elif args.dataset == 'places365':
        dataset_path = constants.PLACES365_PATH
    elif args.dataset == 'imagenet100':
        if args.model in ['resnet18nopt', 'resnet18pt']:
            dataset_path = constants.IMAGENET100_224_PATH
        else:
            dataset_path = constants.IMAGENET100_32_PATH
    
    data = np.load(dataset_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()   
    
    return X_train, y_train, X_test, y_test

def load_model_and_data(args):
    X_train, y_train, X_test, y_test = load_data(args)
    device = args.device
    
    if args.dataset == 'cifar10':
        if args.model == 'alicnn':
            model = AliCNN()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModel()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'rlcnn':
            model = CNNModelRL()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forCifar10(pretrained=False)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        elif args.model == 'resnet18pt':
            model = Resnet18forCifar10(pretrained=True)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        else:
            raise ValueError('Model not found')
    elif args.dataset == 'cifar100':
        if args.model == 'alicnn':
            model = AliCNNforCifar100()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModelforCifar100()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forCifar100(pretrained=False)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        elif args.model == 'resnet18pt':
            model = Resnet18forCifar100(pretrained=True)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        elif args.model == 'densenet121nopt':
            model = Densenet121forCifar100(pretrained=False)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        elif args.model == 'densenet121pt':
            model = Densenet121forCifar100(pretrained=True)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        else:
            raise ValueError('Model not found')
    elif args.dataset == 'imagenet100':
        if args.model == 'alicnn':
            model = AliCNNforImagenet100()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModelforCifar100()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forImagenet100(pretrained=False)
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'resnet18pt':
            model = Resnet18forImagenet100(pretrained=True)
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'rl_alicnn':
            backbone_model = AliCNNforImagenet100()
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=100
            )
            print(model)
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'rl_cnn':
            backbone_model = CNNModelforImagenet100()
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=100
            )
            print(model)
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'rl_resnet18nopt':
            backbone_model = Resnet18forImagenet100(pretrained=False)
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=100
            )
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            train_transform_used = train_transform
            test_transform = val_test_transform
            print(model)
        else:
            raise ValueError('Model not found')
    elif args.dataset == 'places365':
        if args.model == 'alicnn':
            model = AliCNNforPlaces365()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModelforPlaces365()
            train_transform_used = train_transform
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forPlaces365(pretrained=False)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
        elif args.model == 'resnet18pt':
            model = Resnet18forPlaces365(pretrained=True)
            train_transform_used = train_transform_224
            test_transform = val_test_transform_224
    else:
        raise ValueError('Model not found')
    
    if args.model_path:
        model.load_state_dict(torch.load(f'{args.model_path}', map_location=args.device))
    train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform_used)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True             
    )
    test_dataset = CustomImageDataset(X_test, y_test, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True            
    )
    model.to(device)
    return model, train_loader, test_loader

def setup_logging(log_path):
    """
    Configure logging to output to both console and file.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum logging level to INFO
    # If there are already handlers, clear them first. This is especially important in Jupyter environments to prevent duplicate logging.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    # %(asctime)s - time, %(levelname)s - log level, %(message)s - log message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler for writing log files
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler for printing logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def autoattack(args):
    device = args.device

    print("Configuration Setting up...")
    model, train_loader, test_loader = load_model_and_data(args)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()

    print('Logger setting...')
    # Create a unique folder for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("../ae_run", f"{args.model}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, 'run_log.log')
    logger = setup_logging(log_file_path)
    logger.info(f"Trained model will be saved in: {run_dir}")

    print('Auto attack on dataset started...')

    norm = args.norm
    eps = args.eps
    log_path = os.path.join(run_dir, 'aa_log.txt')
    version = args.version 
    state_path =  args.state_path
    batch_size = args.batch_size
    individual = args.individual

    # Load attack
    adversary = AutoAttack(model, norm=norm, eps=eps, log_path=log_path, version=version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0).long() 
    n_ex = len(x_test)
    
    # Example of custom version
    if version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # Run attack and save images
    with torch.no_grad():
        if not individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex], bs=batch_size, state_path=state_path)
    
            torch.save({'adv_complete': adv_complete}, f'{run_dir}/aa_{version}_1_{n_ex}_eps_{eps:.5f}_SS.pth')
    
        else:
            # Individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:n_ex], y_test[:n_ex], bs=batch_size)
    
            torch.save(adv_complete, f'{run_dir}/aa_{version}_individual_1_{n_ex}_eps_{eps:.5f}_SS.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--device', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, required=False, default='cifar10')
    parser.add_argument('--model', type=str, required=False, default='cnn')
    parser.add_argument('--model_path', type=str, required=False, default="../training_runs/cnn/cnn_best_model.pth")
    parser.add_argument('--num_workers', type=int, required=False, default=4)
    parser.add_argument('--norm', type=str, required=False, default='2')
    parser.add_argument('--eps', type=float, required=False, default=1.0)
    parser.add_argument('--version', type=str, required=False, default='standard')
    parser.add_argument('--state_path', type=str, required=False, default=None)
    parser.add_argument('--individual', type=bool, required=False, default=False)
    args = parser.parse_args()
    autoattack(args)