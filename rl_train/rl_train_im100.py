from random import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from data import CustomImageDataset

def load_data():
    data = np.load('data/imagenet100_224.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Split training set: 90% for training, 10% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_data(args):
    """
    Load model and data loaders based on the specified model type.
    
    The validation set is created by splitting 10% from the training set.
    """
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    if args.model == 'resnet':
        from models.resnet import Resnet18forImagenet100
        from data.transform import train_transform_224, val_test_transform_224
        model = Resnet18forImagenet100(pretrained=False)
        train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform_224)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_dataset = CustomImageDataset(X_val, y_val, transform=val_test_transform_224)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_dataset = CustomImageDataset(X_test, y_test, transform=val_test_transform_224)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.model == 'alicnn':
        from models.alicnn import AliCNNforImagenet100
        from data.transform import train_transform, val_test_transform
        model = AliCNNforImagenet100()
        train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_dataset = CustomImageDataset(X_val, y_val, transform=val_test_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_dataset = CustomImageDataset(X_test, y_test, transform=val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.model == 'cnn':
        from models.cnn import CNNModelforImagenet100
        from data.transform import train_transform, val_test_transform
        model = CNNModelforImagenet100()
        train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_dataset = CustomImageDataset(X_val, y_val, transform=val_test_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_dataset = CustomImageDataset(X_test, y_test, transform=val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        raise ValueError('Model not found')
    return model, train_loader, val_loader, test_loader

def load_agent(model, args):
    if args.agent == 'PolicyAdvEpsilonAgent':
        from rl_train.agent.policy_gradient_epsilon import PolicyAdvEpsilonAgent
        device = torch.device(args.device)
        agent = PolicyAdvEpsilonAgent(model, lr=args.learning_rate, device=device, epsilon=args.exploration_epsilon)
    else:
        raise ValueError('Agent not found')

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        agent.model.load_state_dict(state_dict)
    return agent

def train(args):
    # Create a unique folder for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("training_runs", f"{args.model}_{args.agent}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device(args.device)
    model, train_loader, val_loader, test_loader = load_model_and_data(args)
    agent = load_agent(model, args)
    best_val_acc = 0.0
    adv_best_val_acc = 0.0
    log_file_path = os.path.join(run_dir, 'training_log.jsonl')
    model_save_path = os.path.join(run_dir, f'{args.model}_model.pth')
    model_save_path_adv = os.path.join(run_dir, f'{args.model}_adv_model.pth')

    # Save the configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f'Training started. Results will be saved in {run_dir}')

    for epoch in range(args.num_epochs):
        agent.model.train()

        total_train_loss = 0.0
        total_adv_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Update model
            if epoch >= args.start_adv_epoch:
                basic_loss, adv_loss = agent.adv_update(images, labels, args.adv_rate, args.epsilon)
            else:
                basic_loss, adv_loss = agent.update(images, labels)
            
            total_train_loss += basic_loss
            total_adv_loss += adv_loss

        torch.cuda.empty_cache()

        # Validation process
        agent.model.eval()

        val_correct = 0
        val_total = 0
        val_rewards = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} Validation'):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                labels = labels.view(-1)
                actions = agent.select_action(images)
                rewards = (actions == labels).float()

                val_correct += (actions == labels).sum().item()
                val_total += labels.size(0)
                val_rewards += rewards.sum().item()
        val_acc = val_correct / val_total
        val_avg_reward = val_rewards / val_total

        # Testing process (for logging only, not for model selection)
        test_correct = 0
        test_total = 0
        test_rewards = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Epoch {epoch} Testing'):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                labels = labels.view(-1)
                actions = agent.select_action(images)
                rewards = (actions == labels).float()

                test_correct += (actions == labels).sum().item()
                test_total += labels.size(0)
                test_rewards += rewards.sum().item()
        test_acc = test_correct / test_total
        test_avg_reward = test_rewards / test_total

        print(f'Epoch {epoch} Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}.')

        # Save the best models based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(agent.model.state_dict(), model_save_path)
        if epoch > 0 and epoch % args.saving_epoch == 0:
            model_save_path_epoch = os.path.join(run_dir, f'{args.model}_model_{epoch}.pth')
            torch.save(agent.model.state_dict(), model_save_path_epoch)
        if epoch >= args.start_adv_epoch and val_acc > adv_best_val_acc:
            adv_best_val_acc = val_acc
            torch.save(agent.model.state_dict(), model_save_path_adv)

        # Log training, validation and testing metrics
        log_entry = {
            'epoch': epoch,
            'train_loss': total_train_loss,
            'adv_loss': total_adv_loss,
            'val_acc': val_acc,
            'val_reward': val_avg_reward,
            'test_acc': test_acc,
            'test_reward': test_avg_reward
        }
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        del val_correct, val_total, val_rewards, test_correct, test_total, test_rewards
        torch.cuda.empty_cache()

    print(f'Training completed. Results saved in {run_dir}')
    last_model_save_path = os.path.join(run_dir, f'{args.model}_last_model.pth')
    torch.save(agent.model.state_dict(), last_model_save_path)
    print(f'Last model saved in {last_model_save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--start_adv_epoch', type=int, default=50)
    parser.add_argument('--adv_rate', type=float, default=0.2)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--agent', type=str, default='PolicyAdvEpsilonAgent')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--exploration_epsilon', type=float, default=0.1)
    parser.add_argument('--saving_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)