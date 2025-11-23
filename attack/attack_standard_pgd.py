import sys
import logging
import os
import math

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
from models.cnn import CNNModel, CNNModelforCifar100, CNNModelforPlaces365, CNNModelforImagenet100, CNNModelRL
from models.densenet import Densenet121forCifar10, Densenet121forCifar100
from models.resnet import Resnet18forCifar10, Resnet50forCifar10, Resnet18forCifar100, Resnet18forPlaces365, Resnet18forImagenet100

from new_pgd.new_pgd import projected_gradient_descent

import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from datetime import datetime
import argparse
import random
import json

'''
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
'''

def load_data(args):
    if args.dataset == 'cifar10':
        dataset_path = constants.CIFAR10_PATH
    elif args.dataset == 'cifar100':
        dataset_path = constants.CIFAR100_PATH
    elif args.dataset == 'places365':
        dataset_path = constants.PLACES365_PATH
    elif args.dataset == 'imagenet100':
        if args.model in ['resnet18nopt', 'resnet18pt', 'rl_resnet18nopt']:
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
    _, _, X_test, y_test = load_data(args)
    device = args.device
    agent = None
    
    if args.dataset == 'cifar10':
        if args.model == 'alicnn':
            model = AliCNN()
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModel()
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forCifar10(pretrained=False)
            test_transform = val_test_transform_224
        elif args.model == 'resnet18pt':
            model = Resnet18forCifar10(pretrained=True)
            test_transform = val_test_transform_224
        elif args.model == 'rl_cnn':
            model = CNNModelRL()
            test_transform = val_test_transform
        else:
            raise ValueError('Model not found')
    elif args.dataset == 'cifar100':
        if args.model == 'alicnn':
            model = AliCNNforCifar100()
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModelforCifar100()
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forCifar100(pretrained=False)
            test_transform = val_test_transform_224
        elif args.model == 'resnet18pt':
            model = Resnet18forCifar100(pretrained=True)
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
            test_transform = val_test_transform
        elif args.model == 'rl_cnn':
            backbone_model = CNNModelforImagenet100()
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=100
            )
            print(model)
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            test_transform = val_test_transform
        elif args.model == 'rl_resnet18nopt':
            backbone_model = Resnet18forImagenet100(pretrained=False)
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=100
            )
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            test_transform = val_test_transform
            print(model)
        else:
            raise ValueError('Model not found')
    elif args.dataset == 'places365':
        if args.model == 'alicnn':
            model = AliCNNforPlaces365()
            test_transform = val_test_transform
        elif args.model == 'cnn':
            model = CNNModelforPlaces365()
            test_transform = val_test_transform
        elif args.model == 'resnet18nopt':
            model = Resnet18forPlaces365(pretrained=False)
            test_transform = val_test_transform_224
        elif args.model == 'resnet18pt':
            model = Resnet18forPlaces365(pretrained=True)
            test_transform = val_test_transform_224
        elif args.model == 'rl_alicnn':
            backbone_model = AliCNNforPlaces365()
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=365
            )
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            test_transform = val_test_transform
        elif args.model == 'rl_cnn':
            backbone_model = CNNModelforPlaces365()
            model = ConfidenceGatedModelWrapper(
                backbone_model=backbone_model,
                num_classes=365
            )
            agent = AdaptiveConfidenceAgent(model=model, device=device)
            test_transform = val_test_transform
    else:
        raise ValueError('Model not found')
    
    model.load_state_dict(torch.load(f'{args.model_path}', map_location=args.device))
    
    if agent:
        model = CleverhansWrapper(model)
        
    test_dataset = CustomImageDataset(X_test, y_test, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model.to(device)
    model.eval()
    return model, agent, test_loader

def renormalize(x, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(x.device)
    return x * std + mean


def process_for_display(tensor, std, mean):
    """Let a normalized PyTorch image tensor be converted to a displayable Numpy array."""
    img = tensor.clone().detach().cpu().float()
    img = img * std[:, None, None] + mean[:, None, None]
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0)
    return img.numpy()

def save_multiscale_comparison(original_img, perturbed_img, true_label, pred_clean, pred_adv, save_path, target_label=None):
    """
    Create and save a 3x3 grid comparison image showing the attack effects on the same sample at three different scales.
    
    Args:
        original_img (torch.Tensor): Original image tensor (C, H, W), should be full size.
        perturbed_img (torch.Tensor): Adversarial sample tensor (C, H, W), should be full size.
        ... (other parameters same as above) ...
    """
    # Define the three scales we want to display
    # Format: (row title, scale size), size=None means use original size
    scales = [
        ("Original Size", None),
        ("Medium Size (96x96)", 96),
        ("Tiny Size (32x32)", 32)
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"Multi-Scale Analysis of Adversarial Attack\nTrue: {true_label}, Pred (Clean): {pred_clean}, Pred (Adv): {pred_adv}", fontsize=20)

    # Iterate over each scale (each row)
    for i, (title_prefix, size) in enumerate(scales):
        # Prepare the image tensors for the current scale
        orig_t = original_img.clone()
        pert_t = perturbed_img.clone()

        # If a size is specified, resize the images
        if size is not None:
            # interpolate needs a 4D batch input (B, C, H, W), so we add a dimension and then remove it
            orig_t = F.interpolate(orig_t.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
            pert_t = F.interpolate(pert_t.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
            
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        # Convert the processed tensors to displayable Numpy arrays
        original_display = process_for_display(orig_t, std, mean)
        perturbed_display = process_for_display(pert_t, std, mean)
        perturbation_display = np.clip((perturbed_display - original_display + 0.5), 0, 1)
        
        # --- Draw three images in the i-th row ---
        # 1. Original Image
        axes[i, 0].imshow(original_display)
        axes[i, 0].set_title(f"{title_prefix} - Original")
        axes[i, 0].axis('off')
        
        # 2. Perturbation Image
        axes[i, 1].imshow(perturbation_display)
        axes[i, 1].set_title(f"{title_prefix} - Perturbation")
        axes[i, 1].axis('off')
        
        # 3. Adversarial Image
        axes[i, 2].imshow(perturbed_display)
        adv_subtitle = f"Pred: {pred_adv}"
        if target_label is not None: adv_subtitle += f" (Target: {target_label})"
        axes[i, 2].set_title(f"{title_prefix} - Adversarial\n{adv_subtitle}")
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to accommodate the main title
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def save_comparison_image(original_img, perturbed_img, true_label, pred_clean, pred_adv, save_path, target_label=None):
    """
    Create and save a "three-in-one" comparison image containing the original image, perturbation, and adversarial example.

    Args:
        original_img (torch.Tensor): Original image tensor (C, H, W).
        perturbed_img (torch.Tensor): Adversarial sample tensor (C, H, W).
        true_label (int): True label.
        pred_clean (int): Prediction on the clean image.
        pred_adv (int): Prediction on the adversarial sample.
        save_path (str): Full path to save the image (e.g., '.../comparison_3000.png').
        target_label (int, optional): Target label for targeted attacks.
    """
    # --- Core steps: Denormalization and format conversion ---
    # Define mean and std for denormalization (should be the same as used in your DataLoader)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    original_display = process_for_display(original_img, std, mean)
    perturbed_display = process_for_display(perturbed_img, std, mean)
    
    perturbation = perturbed_display - original_display
    perturbation_display = np.clip((perturbation + 0.5), 0, 1)

    # --- Use Matplotlib for plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Plot Original Image
    axes[0].imshow(original_display)
    axes[0].set_title(f"Original Image\nTrue: {true_label}, Pred: {pred_clean}", fontsize=14)
    axes[0].axis('off')

    # 2. Plot Perturbation Image
    axes[1].imshow(perturbation_display)
    axes[1].set_title("Perturbation (Noise)", fontsize=14)
    axes[1].axis('off')

    # 3. Plot Adversarial Image
    title_adv = f"Adversarial Example\nPred: {pred_adv}"
    if target_label is not None:
        title_adv += f" (Target: {target_label})"
    axes[2].imshow(perturbed_display)
    axes[2].set_title(title_adv, fontsize=14, color='red' if pred_adv != true_label else 'green')
    axes[2].axis('off')

    plt.tight_layout()
    # --- Save the image and close the figure to prevent memory leaks ---
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # Must close in a loop, otherwise all images will remain in memory!
    
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

def calculate_entropy(probs):
    """Calculate the entropy of a probability distribution tensor"""
    # Add a small number to prevent log(0)
    # 1e-8: This may cause the calculated value to be slightly different from the original
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)
    return entropy

def generate_and_plot_landscape(image_clean, label, model, agent, device, save_path, args, vmin=0, vmax=20, attack_steps=250):
    attack_norm = args.norm
    attack_eps = args.eps
    # attack_steps = args.nb_iter
       
    with torch.no_grad():
        model.eval()
        if agent:
            agent.model.eval()

        # --- Step 1: Generate adversarial examples to determine the "attack direction" ---
        image_clean_batch = image_clean.unsqueeze(0).to(device)

        # To generate gradients, we need to temporarily enable gradient computation
        with torch.enable_grad():
            image_clean_batch.requires_grad = True
            # Note: The PGD function internally handles gradient computation, we just need to ensure the input allows gradient computation
            if args.is_target:
                print("Target")
                y_target = torch.tensor([label], dtype=torch.long, device=device)
                image_adv_batch = projected_gradient_descent(
                    model, image_clean_batch, eps=attack_eps, eps_iter=attack_eps/10, 
                    nb_iter=attack_steps, norm=np.inf if attack_norm == 'inf' else 2, 
                    y=y_target, targeted=args.is_target
                )
            else:
                print("Non-Target")
                image_adv_batch = projected_gradient_descent(
                    model, image_clean_batch, eps=attack_eps, eps_iter=attack_eps/10,
                    nb_iter=attack_steps, norm=np.inf if attack_norm == 'inf' else 2
                )
        image_adv = image_adv_batch.squeeze(0)

        # --- Step 2: Define the 2D plane for visualization ---
        dir1 = (image_adv - image_clean).cpu()
        dir1_norm = torch.linalg.norm(dir1)
        dir1 = dir1 / dir1_norm if dir1_norm > 0 else dir1

        dir2 = torch.randn_like(dir1)
        dir2 -= torch.dot(dir2.view(-1), dir1.view(-1)) * dir1
        dir2_norm = torch.linalg.norm(dir2)
        dir2 = dir2 / dir2_norm if dir2_norm > 0 else dir2

        # --- Step 3: Create a grid on the 2D plane and compute the value at each point ---
        resolution = 300
        alphas = torch.linspace(-3.0 * attack_eps, 3.0 * attack_eps, resolution)
        gammas = torch.linspace(-3.0 * attack_eps, 3.0 * attack_eps, resolution)

        loss_grid = torch.zeros((resolution, resolution))
        pred_grid = torch.zeros((resolution, resolution))

        criterion = nn.CrossEntropyLoss()
        label_tensor = torch.tensor([label], device=device)

        for i, alpha in enumerate(tqdm(alphas, desc="Visualizing Landscape")):
            for j, gamma in enumerate(gammas):
                perturbation = alpha * dir1 + gamma * dir2
                image_new = image_clean.cpu() + perturbation.cpu()
                image_new = image_new.unsqueeze(0).to(device)

                # Get prediction and loss
                logits = model(image_new)
                pred = logits.argmax(dim=1).item()

                loss = criterion(logits, label_tensor)

                loss_grid[j, i] = loss.item()
                pred_grid[j, i] = pred
                
    # --- Step 3.5: Create a binary "correct/incorrect" grid ---
    # Compare the prediction grid with the true label to get a boolean grid, then convert to float (0.0 for False, 1.0 for True)
    # 1. Ensure true_label is a Python number (int)
    true_label_val = label.item() if torch.is_tensor(label) else label

    # 2. Now, comparing a 2D tensor with a single number allows safe broadcasting
    binary_pred_grid = (pred_grid == true_label_val).float()

    # --- Step 4: Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- Left plot: Loss landscape ---
    contour1 = axes[0].contourf(
        alphas, gammas, loss_grid, 
        levels=20, cmap='magma', 
        vmin=vmin, vmax=vmax
    )
    axes[0].plot(0, 0, 'go', markersize=12, label='Clean Image')
    dir1_norm_val = torch.linalg.norm((image_adv - image_clean).cpu()).item()
    axes[0].plot(dir1_norm_val, 0, 'rX', markersize=12, label='Adversarial Image')
    axes[0].set_title("Loss Landscape", fontsize=16)
    axes[0].set_xlabel("Direction: Adversarial", fontsize=12)
    axes[0].set_ylabel("Direction: Random Orthogonal", fontsize=12)
    axes[0].legend()
    fig.colorbar(contour1, ax=axes[0])

    # --- Right plot: Decision boundary ---
    # a. Create a custom binary colormap: [Color0 (Incorrect), Color1 (Correct)]
    binary_cmap = ListedColormap(['#FFB3BA', '#BAFFC9']) # Light red, light green

    # b. Use the binary grid and colormap for plotting
    #    levels=[-0.5, 0.5, 1.5] ensures 0 maps to the first color, 1 maps to the second
    contour2 = axes[1].contourf(alphas, gammas, binary_pred_grid, 
                                levels=[-0.5, 0.5, 1.5], 
                                cmap=binary_cmap)

    # c. Update title and legend
    axes[1].plot(0, 0, 'go', markersize=12, label=f'Clean Image (True Label: {label})')
    axes[1].plot(dir1_norm_val, 0, 'rX', markersize=12, label='Adversarial Image')
    axes[1].set_title("Decision Boundary (Correct vs. Incorrect)", fontsize=16)
    axes[1].set_xlabel("Direction: Adversarial", fontsize=12)

    # d. Add text annotations for the color blocks
    # Create a legend next to the plot to explain the colors
    legend_patches = [mpatches.Patch(color='#FFB3BA', label='Incorrect Prediction'),
                      mpatches.Patch(color='#BAFFC9', label='Correct Prediction')]
    axes[1].legend(handles=legend_patches, loc='lower right')

    # e. Save the figure
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Visualization image successfully saved to: {save_path}")  

def _safe_json_value(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

def _series_to_iter_dict(seq):
    # 1-based index, keys in the format "iter=1"
    return {f"iter={i+1}": _safe_json_value(val) for i, val in enumerate(seq)}

def save_tracking_json(save_path, step_stats, nb_iter):
    payload = {
        "cos_to_prev": _series_to_iter_dict(step_stats["cos_to_prev"]),           # { "iter=1": null, "iter=2": 0.84, ...}
        "sign_consistency": _series_to_iter_dict(step_stats["sign_consistency"]),
        "grad_norm": _series_to_iter_dict(step_stats["grad_norm"]),
        "dir_var_step": _series_to_iter_dict(step_stats["dir_var_step"]),       
        "dir_var_overall": _safe_json_value(step_stats.get("dir_var", None)),     
        "nb_iter": nb_iter
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)

def attack(args):
    # Create a unique folder for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("../ae_runs", f"{args.dataset}_{args.model}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, 'run_log.log')
    logger = setup_logging(log_file_path)
    logger.info(f"Adversarial examples will be saved in: {run_dir}")
    
    samples_processed = 0
    next_save_milestone = 1000

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Run started. Saving results to: {run_dir}")
    logger.info(f"Attack arguments: {vars(args)}")
    
    logger.info("==================== Initialization ====================")
    # --- Initialization ---
    model, agent, test_loader = load_model_and_data(args)
    device = args.device
    #model.to(device)
    #model.eval()

    # --- Attack params ---
    eps = args.eps
    eps_iter = args.eps_iter
    nb_iter = args.nb_iter
    norm = args.norm

    if norm == "2":
        norm = 2
    elif norm == "np.inf":
        norm = np.inf
    else:
        raise ValueError('Norm order must be either np.inf or 2')

    # --- For evaluation ---
    nb_test = 0
    correct_model = 0
    correct_model_pgd = 0
    if agent:
        correct_agent = 0
        correct_agent_pgd = 0
    if args.is_target:
        # Used to calculate Attack Success Rate (ASR)
        model_asr_count = 0
        agent_asr_count = 0

        # Used to calculate accuracy for the "target class" itself
        target_class_total = 0  # Count how many samples in the test set have the true label as our target class
        target_class_model_correct_clean = 0
        target_class_model_correct_pgd = 0
        if agent:
            target_class_agent_correct_clean = 0
            target_class_agent_correct_pgd = 0


    # --- For saving data ---
    all_x_pgd = []
    all_y = []

    # --- Mean and std for renormalization --- 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # ===== Entropy buckets =====
    total_entropy_clean = 0.0
    total_entropy_pgd = 0.0

    # clean
    ent_clean_correct_sum = 0.0
    ent_clean_wrong_sum   = 0.0
    n_clean_correct = 0
    n_clean_wrong   = 0
    # pgd
    ent_pgd_correct_sum = 0.0
    ent_pgd_wrong_sum   = 0.0
    n_pgd_correct = 0
    n_pgd_wrong   = 0
    
    # Before the loop
    sum_cos_to_prev = None
    sum_sign_consistency = None
    sum_grad_norm = None
    sum_weight = 0
    dir_var_list = []  # dir_var for each batch
        
    logger.info("==================== AE generating ====================")
    if args.is_target:
        for x, y in tqdm(test_loader, desc="Targeted Attacking", unit="batch"):
            x, y = x.to(device), y.to(device)
            current_batch_size = x.size(0)

            y_target = torch.full_like(y, args.y_target)
            x_pgd = projected_gradient_descent(model, x, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm, y=y_target, targeted=args.is_target)

            _, y_pred = model(x).max(1)
            _, y_pred_pgd = model(x_pgd).max(1)
            if agent:
                y_agent_pred = agent.select_action(x)
                y_agent_pred_pgd = agent.select_action(x_pgd)

            # general
            nb_test += y.size(0)
            correct_model += y_pred.eq(y.view_as(y_pred)).sum().item()
            correct_model_pgd += y_pred_pgd.eq(y.view_as(y_pred_pgd)).sum().item()
            if agent:
                correct_agent += y_agent_pred.eq(y.view_as(y_agent_pred)).sum().item()
                correct_agent_pgd += y_agent_pred_pgd.eq(y.view_as(y_agent_pred_pgd)).sum().item()

            # target
            model_asr_count += y_pred_pgd.eq(y_target).sum().item()
            if agent:
                agent_asr_count += y_agent_pred_pgd.eq(y_target).sum().item()
                
            # --- 5. Calculate accuracy for the target class itself ---
            # Create a boolean mask to filter samples whose true label is the target class
            mask = y.eq(args.y_target)

            # If there are samples of the target class in the current batch
            if mask.sum().item() > 0:
                # Accumulate the total number of samples of the target class
                target_class_total += mask.sum().item()

                # Compare "predicted values" and "true values" among the masked samples
                # Calculate for clean samples
                target_class_model_correct_clean += y_pred[mask].eq(y[mask]).sum().item()
                # Calculate for adversarial samples (this measures the robustness of the class)
                target_class_model_correct_pgd += y_pred_pgd[mask].eq(y[mask]).sum().item()

                if agent:
                    target_class_agent_correct_clean += y_agent_pred[mask].eq(y[mask]).sum().item()
                    target_class_agent_correct_pgd += y_agent_pred_pgd[mask].eq(y[mask]).sum().item()
                    
            # --- 2. Check if the save point has been crossed ---
            if samples_processed < next_save_milestone and samples_processed + current_batch_size >= next_save_milestone:

                # a. Calculate the index of the sample to be saved within the current batch
                sample_idx_in_batch = next_save_milestone - samples_processed - 1

                # b. Define the save path and filename
                save_path = os.path.join(run_dir, f"comparison_sample_{next_save_milestone}.png")

                # c. Save the "three-in-one" comparison image
                save_comparison_image(
                    original_img=x[sample_idx_in_batch],
                    perturbed_img=x_pgd[sample_idx_in_batch],
                    true_label=y[sample_idx_in_batch].item(),
                    pred_clean=y_pred[sample_idx_in_batch].item(),
                    pred_adv=y_pred_pgd[sample_idx_in_batch].item(),
                    save_path=save_path, 
                    target_label=args.y_target if args.is_target else None
                )
                logger.info(f"Saved comparison image for sample ~{next_save_milestone} to {save_path}")

                # d. Update the next save point
                next_save_milestone += 1000

            # --- 3. Update the total number of samples processed ---
            samples_processed += current_batch_size
            torch.cuda.empty_cache()

        # --- 6. After the loop ends, calculate and print all results ---
        logger.info("--- Targeted Attack Evaluation Results ---")

        # Number of times recognized as 4
        model_asr = model_asr_count / nb_test if nb_test > 0 else 0
        logger.info(f"Model Prediect on Target[{args.y_target}]: {model_asr:.4f} ({model_asr_count} / {nb_test})")
        if agent:
            agent_asr = agent_asr_count / nb_test if nb_test > 0 else 0
            logger.info(f"Agent Predict on Target[{args.y_target}]: {agent_asr:.4f} ({agent_asr_count} / {nb_test})")
        
        # Check and print the accuracy of the target class itself
        if target_class_total > 0:
            model_clean_acc = target_class_model_correct_clean / target_class_total
            model_robust_acc = target_class_model_correct_pgd / target_class_total
            logger.info(f"Model Clean Accuracy on Target[{args.y_target}]: {model_clean_acc:.4f} ({target_class_model_correct_clean}/{target_class_total})")
            logger.info(f"Model Robust Accuracy on Target[{args.y_target}]: {model_robust_acc:.4f} ({target_class_model_correct_pgd}/{target_class_total})")

            if agent:
                agent_clean_acc = target_class_agent_correct_clean / target_class_total
                agent_robust_acc = target_class_agent_correct_pgd / target_class_total
                logger.info(f"Agent Clean Accuracy on Target[{args.y_target}]: {agent_clean_acc:.4f} ({target_class_agent_correct_clean}/{target_class_total})")
                logger.info(f"Agent Robust Accuracy on Target[{args.y_target}]: {agent_robust_acc:.4f} ({target_class_agent_correct_pgd}/{target_class_total})")
        else:
            logger.info(f"Warning: Test set contains no samples from the target class ({args.y_target}).")
        
        
    else:
        for x, y in tqdm(test_loader, desc="Non-targeted Attacking", unit="batch"):
            x, y = x.to(device), y.to(device)

            # If DataLoader returns a single image (C, H, W), add a batch dimension here
            if x.dim() == 3:
                x = x.unsqueeze(0)      # (C, H, W) -> (1, C, H, W)
                if y.dim() == 0:
                    y = y.unsqueeze(0)  # Scalar -> (1,)
            
            current_batch_size = x.size(0)
            
            x_pgd, step_stats = projected_gradient_descent(model, x, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm,
                                                           clip_min=None, clip_max=None,
                                                           y=y, targeted=False,
                                                           rand_init=True, rand_minmax=None, sanity_checks=True,
                                                           return_stats=True,         
                                                           track_per_sample=False,     
                                                           track_K=16)
            
            # If x_pgd is 3D, add a batch dimension
            if x_pgd.dim() == 3:
                x_pgd = x_pgd.unsqueeze(0)
            
            logits_clean = model(x)
            logits_pgd = model(x_pgd)
            
            _, y_pred = logits_clean.max(1)
            _, y_pred_pgd = logits_pgd.max(1)
            
            probs_clean = F.softmax(logits_clean, dim=1)
            probs_pgd = F.softmax(logits_pgd, dim=1)
            
            ent_clean   = calculate_entropy(probs_clean)  # [B]
            ent_adv     = calculate_entropy(probs_pgd)    # [B]

            # ---- Entropy: Overall/Correct/Wrong ----
            total_entropy_clean += ent_clean.sum().item()
            total_entropy_pgd   += ent_adv.sum().item()

            mask_clean_correct = y_pred.eq(y)
            mask_clean_wrong   = ~mask_clean_correct
            mask_pgd_correct   = y_pred_pgd.eq(y)
            mask_pgd_wrong     = ~mask_pgd_correct

            ent_clean_correct_sum += ent_clean[mask_clean_correct].sum().item()
            ent_clean_wrong_sum   += ent_clean[mask_clean_wrong].sum().item()
            n_clean_correct       += mask_clean_correct.sum().item()
            n_clean_wrong         += mask_clean_wrong.sum().item()

            ent_pgd_correct_sum += ent_adv[mask_pgd_correct].sum().item()
            ent_pgd_wrong_sum   += ent_adv[mask_pgd_wrong].sum().item()
            n_pgd_correct       += mask_pgd_correct.sum().item()
            n_pgd_wrong         += mask_pgd_wrong.sum().item()
            
    
            if agent:
                y_agent_pred = agent.select_action(x)
                y_agent_pred_pgd = agent.select_action(x_pgd)
                
            nb_test += y.size(0)
            correct_model += y_pred.eq(y.view_as(y_pred)).sum().item()
            correct_model_pgd += y_pred_pgd.eq(y.view_as(y_pred_pgd)).sum().item()
            if agent:
                correct_agent += y_agent_pred.eq(y.view_as(y_agent_pred)).sum().item()
                correct_agent_pgd += y_agent_pred_pgd.eq(y.view_as(y_agent_pred_pgd)).sum().item()
                
            # --- Record gradient ---
            T = len(step_stats["cos_to_prev"])
            vec_cos = torch.tensor(step_stats["cos_to_prev"], device='cpu')    # [T]
            vec_sign = torch.tensor(step_stats["sign_consistency"], device='cpu')
            vec_gn   = torch.tensor(step_stats["grad_norm"], device='cpu')

            bsz = x.size(0)
            if sum_cos_to_prev is None:
                sum_cos_to_prev = vec_cos * bsz
                sum_sign_consistency = vec_sign * bsz
                sum_grad_norm = vec_gn * bsz
            else:
                sum_cos_to_prev += vec_cos * bsz
                sum_sign_consistency += vec_sign * bsz
                sum_grad_norm += vec_gn * bsz
            sum_weight += bsz

            dir_var_list.append(step_stats["dir_var"])
                
            # --- 2. Check if the save point has been crossed ---
            if samples_processed < next_save_milestone and samples_processed + current_batch_size >= next_save_milestone:

                # a. Calculate the index of the sample to be saved within the current batch
                sample_idx_in_batch = next_save_milestone - samples_processed - 1

                # b. Define the save path and filename
                save_path = os.path.join(run_dir, f"comparison_sample_{next_save_milestone}.png")

                # c. Save the "three-in-one" comparison image
                save_comparison_image(
                    original_img=x[sample_idx_in_batch],
                    perturbed_img=x_pgd[sample_idx_in_batch],
                    true_label=y[sample_idx_in_batch].item(),
                    pred_clean=y_pred[sample_idx_in_batch].item(),
                    pred_adv=y_pred_pgd[sample_idx_in_batch].item(),
                    save_path=save_path, 
                    target_label=args.y_target if args.is_target else None
                )
                logger.info(f"Saved comparison image for sample ~{next_save_milestone} to {save_path}")

                # d. Update the next save milestone
                next_save_milestone += 1000

            # --- 3. Update the total number of processed samples ---
            samples_processed += current_batch_size
            torch.cuda.empty_cache()
       
    # Gradient 
    mean_cos_curve  = (sum_cos_to_prev / sum_weight).tolist()
    mean_sign_curve = (sum_sign_consistency / sum_weight).tolist()
    mean_gn_curve   = (sum_grad_norm / sum_weight).tolist()
    mean_dir_var    = float(np.nanmean(np.array(dir_var_list, dtype=np.float64)))
    
    print(mean_cos_curve)
    print(mean_sign_curve)
    print(mean_gn_curve)
    print(mean_dir_var)

    save_path = os.path.join(run_dir, "pgd_grad_tracking", "tracking_summary.json")
    save_tracking_json(save_path, step_stats, nb_iter)
    
    # ===== Entropy mean values (all/correct/wrong) =====
    mean_ent_clean_all = total_entropy_clean / nb_test if nb_test > 0 else float('nan')
    mean_ent_pgd_all   = total_entropy_pgd   / nb_test if nb_test > 0 else float('nan')

    mean_ent_clean_correct = (ent_clean_correct_sum / n_clean_correct) if n_clean_correct > 0 else float('nan')
    mean_ent_clean_wrong   = (ent_clean_wrong_sum   / n_clean_wrong)   if n_clean_wrong   > 0 else float('nan')
    mean_ent_pgd_correct   = (ent_pgd_correct_sum   / n_pgd_correct)   if n_pgd_correct   > 0 else float('nan')
    mean_ent_pgd_wrong     = (ent_pgd_wrong_sum     / n_pgd_wrong)     if n_pgd_wrong     > 0 else float('nan')
    
    entropy_json = {
        "counts": {
            "nb_test": nb_test,
            "clean_correct": n_clean_correct, "clean_wrong": n_clean_wrong,
            "pgd_correct":   n_pgd_correct,   "pgd_wrong":   n_pgd_wrong
        },
        "mean_entropy": {
            "clean_all": mean_ent_clean_all,
            "clean_correct": mean_ent_clean_correct,
            "clean_wrong": mean_ent_clean_wrong,
            "pgd_all": mean_ent_pgd_all,
            "pgd_correct": mean_ent_pgd_correct,
            "pgd_wrong": mean_ent_pgd_wrong
        }
    }
    with open(os.path.join(run_dir, "entropy_summary.json"), "w") as f:
        json.dump(entropy_json, f, indent=2)

    # Logging
    logger.info(f"Test accuracy for clean examples: {correct_model / nb_test}")
    logger.info(f"Test accuracy for adversarial examples: {correct_model_pgd / nb_test}")

    logger.info(f"[Entropy] Clean (all):    {mean_ent_clean_all:.6f}")
    logger.info(f"[Entropy] Clean correct: {mean_ent_clean_correct:.6f}  (N={n_clean_correct})")
    logger.info(f"[Entropy] Clean wrong:   {mean_ent_clean_wrong:.6f}    (N={n_clean_wrong})")
    logger.info(f"[Entropy] PGD (all):     {mean_ent_pgd_all:.6f}")
    logger.info(f"[Entropy] PGD correct:   {mean_ent_pgd_correct:.6f}    (N={n_pgd_correct})")
    logger.info(f"[Entropy] PGD wrong:     {mean_ent_pgd_wrong:.6f}      (N={n_pgd_wrong})")

    if agent:
        logger.info(f"Test accuracy for clean examples (with tau): {correct_agent / nb_test}")
        logger.info(f"Test accuracy for adversarial examples (with tau): {correct_agent_pgd / nb_test}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--device', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, required=False, default='cifar10')
    parser.add_argument('--model', type=str, required=False, default='cnn')
    parser.add_argument('--model_path', type=str, required=False, default="../training_runs/cnn/cnn_best_model.pth")
    parser.add_argument('--is_target', type=bool, required=False, default=False)
    parser.add_argument('--y_target', type=int, required=False, default=9999)
    parser.add_argument('--eps', type=float, required=False, default=1.0)
    parser.add_argument('--eps_iter', type=float, required=False, default=0.1)
    parser.add_argument('--nb_iter', type=int, required=False, default=100)
    parser.add_argument('--norm', type=str, required=False, default='2')
    parser.add_argument('--loss_target', type=int, required=False, default=0)
    args = parser.parse_args()
    attack(args)