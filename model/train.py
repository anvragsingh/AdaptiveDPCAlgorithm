#!/usr/bin/env python3
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import ResNet50, QuantizedResNet50
from datasets import prepare_datasets
from adpc import AdaptiveDPC
from logger import ExperimentLogger
from utils import evaluate_model, compute_computation_savings

def main():
    parser = argparse.ArgumentParser(description='Train models with Adaptive DPC')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Dataset to use for training')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'quant_resnet50'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    parser.add_argument('--initial-bits', type=int, default=8,
                       help='Initial bit-width for quantization')
    parser.add_argument('--min-bits', type=int, default=4,
                       help='Minimum bit-width for quantization')
    parser.add_argument('--delta', type=float, default=0.5,
                       help='Precision interval parameter for DPC')
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive precision adjustment')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--experiment-name', type=str, default='experiment',
                       help='Name for the experiment')
    args = parser.parse_args()

    # Prepare datasets
    train_loader, val_loader, test_loader, num_classes = prepare_datasets(
        args.dataset, batch_size=args.batch_size)

    # Create model
    if args.model == 'resnet50':
        model = ResNet50(num_classes=num_classes).to(args.device)
    else:
        model = QuantizedResNet50(num_classes=num_classes, 
                                 initial_bits=args.initial_bits,
                                 min_bits=args.min_bits).to(args.device)

    # Initialize Adaptive DPC
    adpc = AdaptiveDPC(model, args.initial_bits, args.min_bits, args.delta)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Experiment logger
    logger = ExperimentLogger(args.experiment_name)

    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()

            # Quantize weights before forward pass
            if args.adaptive or args.model == 'quant_resnet50':
                adpc.quantize_weights()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update precision periodically for adaptive DPC
            if args.adaptive and batch_idx % 100 == 0:
                gradient_stats = {}
                for name, param in model.named_parameters():
                    if 'weight' in name and param.grad is not None:
                        layer_name = name.replace('.weight', '')
                        grad_norm = param.grad.norm().item()
                        grad_var = param.grad.var().item()
                        gradient_stats[layer_name] = {'norm': grad_norm, 'var': grad_var}
                
                adpc.update_precision(gradient_stats)
                logger.log_bit_width_distribution(adpc, epoch * len(train_loader) + batch_idx)

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, args.device, adpc)
        
        # Compute metrics
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        comp_savings = compute_computation_savings(model, adpc)
        epoch_time = time.time() - start_time

        # Log metrics
        metrics = {
            'train/loss': avg_train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'computation/savings': comp_savings,
            'time/epoch': epoch_time
        }
        logger.log_metrics(metrics, epoch)
        logger.log_model_parameters(model, epoch)

        # Update learning rate
        scheduler.step()

        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Time: {epoch_time:.2f}s | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'Savings: {comp_savings:.2f}%')

    # Final test evaluation
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, args.device, adpc)
    logger.log_metrics({'test/loss': test_loss, 'test/accuracy': test_acc}, args.epochs)
    
    print(f'\nFinal Test Results: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%')

    # Save experiment results
    logger.save_experiment()
    
    # Save model
    torch.save(model.state_dict(), f'{args.experiment_name}_model.pth')

if __name__ == '__main__':
    main()