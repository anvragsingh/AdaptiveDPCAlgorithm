import torch
import numpy as np
from collections import defaultdict
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def prepare_datasets(dataset_name, batch_size=128, val_ratio=0.1):
    """
    Prepare CIFAR-10, CIFAR-100, or ImageNet datasets with augmentation and normalization
    """
    # Normalization parameters
    normalize_cifar = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    normalize_imagenet = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if dataset_name in ['cifar10', 'cifar100']:
        # Common CIFAR transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_cifar
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_cifar
        ])
        
        # Load datasets
        dataset_class = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100
        train_set = dataset_class(
            root='./data', train=True, download=True, transform=train_transform)
        test_set = dataset_class(
            root='./data', train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'imagenet':
        # ImageNet transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_imagenet
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_imagenet
        ])
        
        # Note: ImageNet requires manual download
        data_path = './data/imagenet'
        train_set = datasets.ImageFolder(
            os.path.join(data_path, 'train'), transform=train_transform)
        test_set = datasets.ImageFolder(
            os.path.join(data_path, 'val'), transform=test_transform)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split training set into train and validation
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(val_ratio * num_train)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(train_set, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, criterion, device, adpc=None):
    """Evaluate model on a given dataset"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if adpc is not None:
                adpc.quantize_weights()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def compute_computation_savings(model, adpc, baseline_bits=32):
    """Compute computational savings from quantization"""
    total_ops = 0
    quantized_ops = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.replace('.weight', '')
            bits = adpc.bit_widths.get(layer_name, baseline_bits)
            
            # Simplified computation: ops proportional to bit-width
            layer_ops = param.numel() * baseline_bits
            quantized_layer_ops = param.numel() * bits
            
            total_ops += layer_ops
            quantized_ops += quantized_layer_ops
    
    savings = 100 * (1 - quantized_ops / total_ops)
    return savings

def save_checkpoint(model, optimizer, adpc, epoch, filename):
    """Save training checkpoint"""
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'adpc_state': adpc.bit_widths
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, adpc, filename):
    """Load training checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    adpc.bit_widths = checkpoint['adpc_state']
    return checkpoint['epoch']

def count_parameters(model):
    """Count total number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_layer_capacities(model):
    """Calculate capacity (number of parameters) for each layer"""
    capacities = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.replace('.weight', '')
            capacities[layer_name] = param.numel()
    return capacities