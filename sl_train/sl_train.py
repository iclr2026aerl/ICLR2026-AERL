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
from models.cnn import CNNModel, CNNModelforCifar100, CNNModelforPlaces365, CNNModelforImagenet100
from models.densenet import Densenet121forCifar10, Densenet121forCifar100
from models.resnet import Resnet18forCifar10, Resnet50forCifar10, Resnet18forCifar100, Resnet18forPlaces365, Resnet18forImagenet100
from models.normalization import NormalizationWrapper

import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from datetime import datetime
import argparse
import random
import json

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


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
    
    # Split training set: 90% for training, 10% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_model_and_data(args):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args)
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

    # Wrap the model with normalization
    model = NormalizationWrapper(model, args.dataset, args.device)
    
    train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform_used)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True             
    )
    val_dataset = CustomImageDataset(X_val, y_val, transform=test_transform)
    val_loader = DataLoader(
        val_dataset, 
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
    return model, train_loader, val_loader, test_loader

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
    # %(asctime)s - Time, %(levelname)s - Log level, %(message)s - Log message
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fgsm_attack(image, epsilon, gradient):
    # Collect the element-wise sign of the gradient
    sign_grad = gradient.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def train_one_epoch(model, dataloader, criterion, optimizer, device, current_epoch, start_adv_epoch, epsilon, beta_adv=1.0, clip_value=1.0):
    """
    A complete single-epoch training function with conditional adversarial training.
    Standard training is performed before start_adv_epoch, followed by TRADES-style adversarial training.
    """
    model.train()
    running_loss = 0.0
    
    # Decide the description text for Tqdm based on the current epoch
    desc = f"Epoch {current_epoch+1} [Clean Training]" if current_epoch < start_adv_epoch else f"Epoch {current_epoch+1} [Adv. Training]"
    progress_bar = tqdm(dataloader, desc=desc)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # --- Core decision logic ---
        if current_epoch < start_adv_epoch:
            # --- Branch 1: Standard clean training ---
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
        else:
            # --- Branch 2: TRADES-style adversarial training ---
            images.requires_grad = True

            # 1. Calculate the loss on clean samples
            outputs_clean = model(images)
            loss_clean = criterion(outputs_clean, labels)
            
            # 2. Generate adversarial samples
            grad = torch.autograd.grad(outputs=loss_clean, inputs=images,
                                       grad_outputs=torch.ones_like(loss_clean),
                                       retain_graph=True)[0]
            images_adv = fgsm_attack(images.detach(), epsilon, grad.detach())
            
            # 3. Calculate the loss on adversarial samples
            outputs_adv = model(images_adv)
            loss_adv = criterion(outputs_adv, labels)
            
            # 4. Combine the losses
            loss = loss_clean + beta_adv * loss_adv
            loss.backward()

        # --- Common update steps ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{running_loss / (progress_bar.n + 1):.4f}")
        
    return running_loss / len(dataloader)

def lambda_lr(epoch):
    '''
    Use the warmup and decay learning rate strategy
    '''
    warmup_epochs = 5
    max_epochs = 100
    min_lr_ratio = 0.1

    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        return max(min_lr_ratio, float(max_epochs - epoch) / float(max(1, max_epochs - warmup_epochs)))
    
def train(args):

    num_epochs = args.num_epochs
    start_adv_epoch = args.start_adv_epoch
    device = args.device
    epsilon = args.epsilon
    beta_adv = args.beta_adv
    clip_value = args.clip_value

    if start_adv_epoch > num_epochs or start_adv_epoch < 0:
        start_adv_epoch = num_epochs + 1 # means never use adversarial training

    # Create a unique folder for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("../training_runs", f"{args.model}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, 'run_log.log')
    logger = setup_logging(log_file_path)
    logger.info(f"Trained model will be saved in: {run_dir}")

    # get the training pre-requisites
    model, train_loader, val_loader, test_loader = load_model_and_data(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # Logging configuration
    best_acc = 0.0
    log_file_path = os.path.join(run_dir, 'training_log.jsonl')
    best_model_save_path = os.path.join(run_dir, f'{args.model}_best_model.pth')
    final_model_save_path = os.path.join(run_dir, f'{args.model}_final_model.pth')
    

    # Save the configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f'Training started. Results will be saved in {run_dir}')

    for epoch in range(num_epochs):
        # Training loop
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, current_epoch=epoch, start_adv_epoch=start_adv_epoch, epsilon=epsilon, beta_adv=beta_adv, clip_value=clip_value)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            val_acc = val_correct / val_total

        # Testing loop (for logging only, not for model selection)
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            test_acc = test_correct / test_total

        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            if isinstance(model, NormalizationWrapper):
                torch.save(model.model.state_dict(), best_model_save_path)
            else:
                torch.save(model.state_dict(), best_model_save_path)
        
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc,
            'test_loss': test_loss / len(test_loader),
            'test_acc': test_acc,
        }
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        logger.info(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100 * val_acc:.2f}%, Test Accuracy: {100 * test_acc:.2f}%')

        scheduler.step()
        
    # Save final model
    if isinstance(model, NormalizationWrapper):
        torch.save(model.model.state_dict(), final_model_save_path)
    else:
        torch.save(model.state_dict(), final_model_save_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--num_epochs', type=int, required=True, default=100)
    parser.add_argument('--device', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0002)
    parser.add_argument('--dataset', type=str, required=False, default='cifar10')
    parser.add_argument('--model', type=str, required=False, default='cnn')
    parser.add_argument('--beta_adv', type=float, required=False, default=1.0)
    parser.add_argument('--epsilon', type=float, required=False, default=0.01)
    parser.add_argument('--clip_value', type=float, required=False, default=1.0)
    parser.add_argument('--model_path', type=str, required=False, default=None)
    parser.add_argument('--start_adv_epoch', type=int, required=False, default=-1, help='Epoch to start adversarial training, -1 means no adversarial training')
    parser.add_argument('--num_workers', type=int, required=False,default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
    