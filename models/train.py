from os import path
from typing import List

import numpy as np
import torchvision.transforms

from .models import CNNClassifier, load_model, save_model
from .utils import load_data, accuracy
import torch.utils.tensorboard as tb
import torch


def train(args):
    """
    Method that trains a given model
    :param args: ArgumentParser with args to run the training (goto main to see the options)
    """
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (args.cpu or args.debug) else 'cpu')
    print(device)

    # Number of epoch after which to validate and save model
    steps_validate = 10

    # Hyperparameters
    lrs: List[int] = args.lrs
    optimizers: List[str] = ["adam"]  # adam, sgd
    n_epochs: int = args.n_epochs
    batch_size: int = args.batch_size
    num_workers: int = 0 if args.debug else args.num_workers
    dimensions: List[List[int]] = [[32, 64, 128, 256], [32, 64, 128]]
    scheduler_modes = ['min_loss', 'max_acc', 'max_val_acc']  # min_loss, max_acc, max_val_acc
    residual: bool = not args.non_residual
    max_pooling: bool = not args.non_max_pooling

    model = None
    loss = torch.nn.BCEWithLogitsLoss()
    transforms = {
        "h": torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()]),
        "h128": torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                torchvision.transforms.RandomHorizontalFlip()]),
        "h_cj9": torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                 torchvision.transforms.ColorJitter(0.9, 0.9, 0.9, 0.1)]),
        "h_cj5": torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                 torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)])
    }

    for t, transform in transforms.items():
        # load train and test data
        loader_train, loader_valid, _ = load_data(f"{args.data_path}/UTKFace", num_workers=num_workers,
                                                  batch_size=batch_size, transform=transform, lengths=(0.7, 0.15, 0.15))

        for optim in optimizers:
            for s_mode in scheduler_modes:
                for dim in dimensions:
                    for lr in lrs:
                        # Tensorboard
                        global_step = 0
                        best_val_acc = 0
                        name_model = f"{optim}/{s_mode}/{batch_size}/{dim}/{lr}/" \
                                     f"residual={residual}/maxPool={max_pooling}"
                        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train', name_model), flush_secs=1)
                        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid', name_model), flush_secs=1)

                        del model
                        model = CNNClassifier(in_channels=3, out_channels=1, dim_layers=dim, residual=residual,
                                              max_pooling=max_pooling).to(device)

                        if optim == "sgd":
                            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                        elif optim == "adam":
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        else:
                            raise Exception("Optimizer not configured")

                        if s_mode == "min_loss":
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
                        elif s_mode == "max_acc" or s_mode == "max_val_acc":
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
                        else:
                            raise Exception("Optimizer not configured")

                        print(f"{args.log_dir}/{name_model}")
                        for epoch in range(n_epochs):
                            print(epoch)
                            train_loss = []
                            train_acc = []

                            # Start training
                            model.train()
                            for img, _, gender in loader_train:
                                # To device
                                img = img.to(device)
                                # age = age.to(device)
                                gender = gender.to(device)

                                # Compute loss and update parameters
                                pred = model(img).squeeze(1)
                                loss_val = loss(pred, gender)
                                optimizer.zero_grad()
                                loss_val.backward()
                                optimizer.step()

                                # Add train loss and accuracy
                                train_loss.append(loss_val.cpu().detach().numpy())
                                train_acc.append(accuracy(pred, gender))

                            # Evaluate the model
                            val_acc = []
                            model.eval()
                            for img, _, gender in loader_valid:
                                # To device
                                img = img.to(device)
                                # age = age.to(device)
                                gender = gender.to(device)

                                val_acc.append(accuracy(model(img).squeeze(1), gender))

                            train_loss = np.mean(train_loss)
                            train_acc = np.mean(train_acc)
                            val_acc = np.mean(val_acc)

                            # Step the scheduler to change the learning rate
                            if s_mode == "min_loss":
                                scheduler.step(train_loss)
                            elif s_mode == "max_acc":
                                scheduler.step(train_acc)
                            elif s_mode == "max_val_acc":
                                scheduler.step(val_acc)

                            global_step += 1
                            if train_logger is not None:
                                train_logger.add_scalar('loss', train_loss, global_step=global_step)
                                train_logger.add_scalar('acc', train_acc, global_step=global_step)
                                valid_logger.add_scalar('acc', val_acc, global_step=global_step)
                                train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                                        global_step=global_step)

                            # Save the model
                            if (epoch % steps_validate == steps_validate - 1) and val_acc > best_val_acc:
                                save_model(model, f"models/{name_model}".replace('/', '_'))
                                best_val_acc = val_acc


def validate(args, metric_func):
    """
    Calculates the validation metric of the model given in args
    :param args: ArgumentParser with args to run the training (goto main to see the options)
    :param metric_func: function that calculates a metric for validation
    :return: validation result according to a the metric_func given
    """
    # todo do the validate function
    pass


if __name__ == '__main__':
    from argparse import ArgumentParser

    args_parser = ArgumentParser()

    args_parser.add_argument('--log_dir', default="./logs")
    args_parser.add_argument('--data_path', default="./data")

    # Hyper-parameters
    args_parser.add_argument('-lrs', nargs='+', type=float, default=[1e-3], help='learning rates')
    # args_parser.add_argument('-opt', '--optimizers', type=str, nargs='+', default=["adam"], help='optimizer to use')
    args_parser.add_argument('-n', '--n_epochs', default=150, type=int, help='number of epochs to train on')
    args_parser.add_argument('-b', '--batch_size', default=64, type=int, help='size of batches to use')
    args_parser.add_argument('-w', '--num_workers', default=2, type=int,
                             help='number of workers to use for data loading')
    args_parser.add_argument('--non_residual', action='store_true',
                             help='if present it will not use residual connections')
    args_parser.add_argument('--non_max_pooling', action='store_true',
                             help='if present the model will not use max pooling (stride in convolutions instead)')

    args_parser.add_argument('--cpu', action='store_true')
    args_parser.add_argument('-d', '--debug', action='store_true')

    args = args_parser.parse_args()
    train(args)
