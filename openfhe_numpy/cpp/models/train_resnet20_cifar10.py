#!/usr/bin/env python3
"""
Train ResNet-20 (no BatchNorm) on CIFAR-10 and export weights for OpenFHE inference.

Architecture (He et al., 2015 adapted for CIFAR-10, without BatchNorm):
  conv1: 3x3, 3->16, stride=1, padding=1
  ReLU
  Layer1: 3 BasicBlocks (16->16, stride=1)
  Layer2: 3 BasicBlocks (16->32, first stride=2)
  Layer3: 3 BasicBlocks (32->64, first stride=2)
  Global AvgPool 8x8
  FC: 64->10

Each BasicBlock:
  conv1: 3x3, stride=s, padding=1 -> ReLU -> conv2: 3x3, stride=1, padding=1
  + shortcut (identity or 1x1 conv for dimension change)
  -> ReLU

Usage:
  python train_resnet20_cifar10.py                    # Train from scratch
  python train_resnet20_cifar10.py --epochs 300       # Custom epochs
  python train_resnet20_cifar10.py --export-only model.pt  # Export existing model
"""

import argparse
import os
import sys
import struct
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ==================== Model Definition ====================

class BasicBlock(nn.Module):
    """ResNet BasicBlock without BatchNorm."""

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=True),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10 without BatchNorm."""

    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

        # Kaiming initialization (important without BatchNorm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==================== Training ====================

def train_model(epochs=300, lr=0.01, batch_size=128, data_dir='./data',
                save_path='resnet20_cifar10.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Cosine annealing schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total

        # Test
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"Train Loss: {train_loss/len(trainloader):.4f}  "
                  f"Train Acc: {train_acc:.2f}%  "
                  f"Test Acc: {test_acc:.2f}%  "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    return model


# ==================== Weight Export ====================

def save_tensor(tensor, bin_path, shape_path):
    """Save a tensor as .bin (float64) and .shape files."""
    data = tensor.detach().cpu().numpy().astype(np.float64)
    with open(bin_path, 'wb') as f:
        f.write(data.tobytes())
    with open(shape_path, 'w') as f:
        f.write(' '.join(str(d) for d in data.shape))


def export_weights(model_path, output_dir):
    """Export PyTorch model weights to .bin/.shape format for C++ inference."""
    os.makedirs(output_dir, exist_ok=True)

    model = ResNet20()
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"Exporting weights to {output_dir}/")

    # Initial conv
    save_tensor(model.conv1.weight, f"{output_dir}/conv1.weight.bin",
                f"{output_dir}/conv1.weight.shape")
    save_tensor(model.conv1.bias, f"{output_dir}/conv1.bias.bin",
                f"{output_dir}/conv1.bias.shape")
    print(f"  conv1.weight: {list(model.conv1.weight.shape)}")

    # Layers 1-3, blocks 0-2
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3], start=1):
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}.{block_idx}"

            save_tensor(block.conv1.weight,
                        f"{output_dir}/{prefix}.conv1.weight.bin",
                        f"{output_dir}/{prefix}.conv1.weight.shape")
            save_tensor(block.conv1.bias,
                        f"{output_dir}/{prefix}.conv1.bias.bin",
                        f"{output_dir}/{prefix}.conv1.bias.shape")
            print(f"  {prefix}.conv1.weight: {list(block.conv1.weight.shape)}")

            save_tensor(block.conv2.weight,
                        f"{output_dir}/{prefix}.conv2.weight.bin",
                        f"{output_dir}/{prefix}.conv2.weight.shape")
            save_tensor(block.conv2.bias,
                        f"{output_dir}/{prefix}.conv2.bias.bin",
                        f"{output_dir}/{prefix}.conv2.bias.shape")
            print(f"  {prefix}.conv2.weight: {list(block.conv2.weight.shape)}")

            # Shortcut conv (only for first block of layer2 and layer3)
            if len(block.shortcut) > 0:
                sc = block.shortcut[0]
                save_tensor(sc.weight,
                            f"{output_dir}/{prefix}.shortcut.weight.bin",
                            f"{output_dir}/{prefix}.shortcut.weight.shape")
                save_tensor(sc.bias,
                            f"{output_dir}/{prefix}.shortcut.bias.bin",
                            f"{output_dir}/{prefix}.shortcut.bias.shape")
                print(f"  {prefix}.shortcut.weight: {list(sc.weight.shape)}")

    # FC layer
    save_tensor(model.fc.weight, f"{output_dir}/fc.weight.bin",
                f"{output_dir}/fc.weight.shape")
    save_tensor(model.fc.bias, f"{output_dir}/fc.bias.bin",
                f"{output_dir}/fc.bias.shape")
    print(f"  fc.weight: {list(model.fc.weight.shape)}")

    print("Weight export complete!")


# ==================== Test Image Export ====================

def export_test_images(num_images=10, data_dir='./data', output_dir=None):
    """Export normalized CIFAR-10 test images as .bin files for C++ inference."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cifar10')
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"\nExporting {num_images} CIFAR-10 test images to {output_dir}/")

    for i in range(min(num_images, len(testset))):
        img, label = testset[i]
        # img shape: [3, 32, 32], already normalized
        data = img.numpy().astype(np.float64)  # CHW format

        filename = f"cifar10_{i}_label_{label}.bin"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(data.tobytes())

        # Also save as text for debugging
        txt_filename = f"cifar10_{i}_label_{label}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        with open(txt_filepath, 'w') as f:
            flat = data.flatten()
            for j, val in enumerate(flat):
                f.write(f"{val:.6f}")
                if j < len(flat) - 1:
                    f.write(' ')

        print(f"  Sample {i}: label={label} ({classes[label]}), "
              f"shape={list(data.shape)}, "
              f"range=[{data.min():.3f}, {data.max():.3f}]")

    print("Image export complete!")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-20 on CIFAR-10 (no BatchNorm)')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-10 data directory')
    parser.add_argument('--save-path', type=str, default='resnet20_cifar10.pt',
                        help='Path to save trained model')
    parser.add_argument('--export-only', type=str, default=None,
                        help='Skip training, export weights from this model file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Weight export directory (default: resnet20_weight_relu/)')
    parser.add_argument('--num-test-images', type=int, default=10,
                        help='Number of test images to export')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, only export from save-path')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'resnet20_weight_relu')

    model_path = args.save_path

    if args.export_only:
        model_path = args.export_only
    elif not args.skip_training:
        print("=" * 60)
        print("  ResNet-20 CIFAR-10 Training (No BatchNorm)")
        print("=" * 60)
        train_model(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            save_path=args.save_path,
        )
        model_path = args.save_path

    # Export weights
    print("\n" + "=" * 60)
    print("  Exporting Weights")
    print("=" * 60)
    export_weights(model_path, args.output_dir)

    # Export test images
    print("\n" + "=" * 60)
    print("  Exporting Test Images")
    print("=" * 60)
    image_output_dir = os.path.join(script_dir, '..', 'data', 'cifar10')
    export_test_images(args.num_test_images, args.data_dir, image_output_dir)

    print("\nDone! Files ready for C++ inference.")


if __name__ == '__main__':
    main()
