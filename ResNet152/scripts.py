import torch

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.mnist import MNIST

import argparse

from train import Train


def main(
    train_path,
    val_path,
    log_path,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    lr_scheduler,
    momentum,
    optimizer,
    num_workers,
    device,
    use_tensorboard,
    checkpoint,
    mnist,
    transfer_learning
    ):


    # Transformation
    if transfer_learning:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.1307]*3, std=[0.3081]*3),
            T.Resize((28, 28), antialias=True),
            ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((28, 28), antialias=True),
            ])


    # Dataset
    if mnist:
        ds_train = MNIST(root='ResNet152/datasets', train=True, transform=transform)
        ds_val = MNIST(root='ResNet152/datasets', train=False, transform=transform)
    else:
        ds_train = ImageFolder(root=train_path, transform=transform)
        ds_val = ImageFolder(root=val_path, transform=transform)

    with open('ResNet152/class_idx.txt', 'w') as f:
        for idx in ds_train.class_to_idx:
            f.writelines('{}: {}\n'.format(idx, ds_train.class_to_idx[idx]))


    # Train
    train = Train(
                ds_train, 
                ds_val,
                log_path,
                num_classes,
                epochs,
                batch_size,
                learning_rate,
                lr_scheduler,
                momentum,
                optimizer,
                num_workers,
                device,
                use_tensorboard,
                checkpoint,
                transfer_learning
                )
    
    train.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', '-tp', help='Train folder', type=str, default='ResNet152/datasets/numbers/train')
    parser.add_argument('--val_path', '-vp', help='Val folder', type=str, default='ResNet152/datasets/numbers/val')
    parser.add_argument('--log_path', '-lp', help='Log folder', type=str, default='ResNet152/logs')
    parser.add_argument('--num_classes', '-c', help='Number of classes', type=int, default=10)
    parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=100) 
    parser.add_argument('--batch_size', '-b', help='Training batch size', type=int, default=128)
    parser.add_argument('--learning_rate', '-lr', help='Specify learning rate', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', '-lrs', help='Specify schedule', type=int, nargs='*', action='store', default=None)
    parser.add_argument('--momentum', '-m', help='Momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', '-o', help='Specify optimizer: SGD, Adam, RMSprop', type=str, default='SGD')
    parser.add_argument('--num_workers', '-w', help='Number of workers', type=int, default=8)
    parser.add_argument('--device', '-d', help='Device', type=str, default=None)
    parser.add_argument('--use_tensorboard', '-tb', help='Use tensorboard', type=bool, default=True)
    parser.add_argument('--checkpoint', '-cp', help='continue from checkpoint', type=str, default=None)
    parser.add_argument('--mnist', help="Use MNIST Dataset", default=True)
    parser.add_argument('--transfer_learning', help="Activate transfer learning", type=str, default=None)
    args = parser.parse_args()

    main(**args.__dict__)