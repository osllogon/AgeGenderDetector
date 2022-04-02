from os import path
from typing import List

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torchvision.transforms
from PIL import Image

from .models import CNNClassifier, load_model, save_model, load_model_from_name
from .utils import load_data, accuracy, save_dict, load_dict, IMAGE_TRANSFORM


def train(args):
    """
    Method that trains a given model
    :param args: ArgumentParser with args to run the training (goto main to see the options)
    """
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (args.cpu or args.debug) else 'cpu')
    print(device)

    # Number of epoch after which to validate and save model
    steps_validate = 1

    # Hyperparameters
    # learning rates
    lrs: List[int] = args.lrs
    # optimizer to use for training
    optimizers: List[str] = ["adam"]  # adam, sgd
    # number of epochs to train on
    n_epochs: int = args.n_epochs
    # size of batches to use
    batch_size: int = args.batch_size
    # number of workers (processes) to use for data loading
    num_workers: int = 0 if args.debug else args.num_workers
    # dimensions of the model to use (look at model for more detail)
    dimensions: List[List[int]] = [[32, 64, 128]]
    # scheduler mode to use for the learning rate scheduler
    scheduler_modes = ['min_mse']  # min_loss, max_acc, max_val_acc, (age) min_mse
    # whether to use residual connections
    residual: bool = not args.non_residual
    # whether to use max pooling instead of stride in convolutions
    max_pooling: bool = not args.non_max_pooling
    # # whether to use flatten instead of mean pooling before the output linear layer
    # flatten_out_layer: bool = args.flatten_out_layer

    # For age and gender model
    # weight for the age loss
    loss_age_weights: List[float] = args.loss_age_weight if args.age_gender else [0.0]

    model = None
    loss_gender = torch.nn.BCEWithLogitsLoss().to(device)  # sigmoid + BCELoss (good for 2 classes classification)
    loss_age = torch.nn.MSELoss().to(device)
    # transforms to use for data augmentation
    transforms = {
        # "none": None,
        "h": torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()]),
        # "h128": torchvision.transforms.Compose([torchvision.transforms.Resize((128,128)),
        #                                         torchvision.transforms.RandomHorizontalFlip()]),
        # "h_cj9": torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
        #                                          torchvision.transforms.ColorJitter(0.9, 0.9, 0.9, 0.1)]),
    }

    # training loop
    for age_weight in loss_age_weights:
        for t, transform in transforms.items():
            # load train and test data
            loader_train, loader_valid, _ = load_data(f"{args.data_path}", num_workers=num_workers,
                                                      batch_size=batch_size, transform=transform,
                                                      drop_last=True, lengths=(0.7, 0.15, 0.15))

            for optim in optimizers:
                for s_mode in scheduler_modes:
                    for dim in dimensions:
                        for lr in lrs:
                            # Tensorboard
                            global_step = 0
                            name_model = f"{optim}/{t}/{s_mode}/{batch_size}/{dim}/{lr}/" \
                                         f"residual={residual}/maxPool={max_pooling}"
                            if args.age_gender:
                                name_model = f"{age_weight}/{name_model}"
                            train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train', name_model), flush_secs=1)
                            valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid', name_model), flush_secs=1)

                            del model
                            dict_model = {
                                # dictionary with model information
                                "name": name_model,
                                "in_channels": 3,
                                "out_channels": 2 if args.age_gender else 1,
                                "dim_layers": dim,
                                "block_conv_layers": 3,
                                "residual": residual,
                                "max_pooling": max_pooling,
                                "acc_gender": 0,
                                "mse_age": 0,
                                "epoch": 0,
                            }
                            model = CNNClassifier(**dict_model).to(device)

                            if optim == "sgd":
                                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                            elif optim == "adam":
                                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                            else:
                                raise Exception("Optimizer not configured")

                            if s_mode == "min_loss" or "min_mse":
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
                            elif s_mode == "max_acc" or s_mode == "max_val_acc":
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
                            else:
                                raise Exception("Optimizer not configured")

                            print(f"{args.log_dir}/{name_model}")
                            for epoch in range(n_epochs):
                                print(epoch)
                                train_loss_gender = []
                                train_loss_age = []
                                train_loss = []
                                train_acc_gender = []

                                # Start training
                                model.train()
                                for img, age, gender in loader_train:
                                    # To device
                                    img, age, gender = img.to(device), age.to(device), gender.to(device)

                                    # Compute loss and update parameters
                                    pred = model(img)
                                    loss_val_gender = loss_gender(pred[:, 0], gender)
                                    loss_val = loss_val_gender
                                    if args.age_gender:
                                        loss_val_age = loss_age(pred[:, 1], age)
                                        loss_val += age_weight * loss_val_age

                                    # Do back propagation
                                    optimizer.zero_grad()
                                    loss_val.backward()
                                    optimizer.step()

                                    # Add train loss and accuracy
                                    train_loss.append(loss_val.cpu().detach().numpy())
                                    train_acc_gender.append(accuracy(pred[:, 0], gender))
                                    if args.age_gender:
                                        train_loss_age.append(loss_val_age.cpu().detach().numpy())
                                        train_loss_gender.append(loss_val_gender.cpu().detach().numpy())

                                # Evaluate the model
                                val_acc_gender = []
                                val_mse_age = []
                                model.eval()
                                for img, age, gender in loader_valid:
                                    # To device
                                    img, age, gender = img.to(device), age.to(device), gender.to(device)
                                    pred = model(img)

                                    val_acc_gender.append(accuracy(pred[:, 0], gender))
                                    if args.age_gender:
                                        val_mse_age.append(loss_age(pred[:, 1], age).cpu().detach().numpy())

                                train_loss = np.mean(train_loss)
                                train_acc_gender = np.mean(train_acc_gender)
                                val_acc_gender = np.mean(val_acc_gender)
                                if args.age_gender:
                                    train_loss_gender = np.mean(train_loss_gender)
                                    train_loss_age = np.mean(train_loss_age)
                                    val_mse_age = np.mean(val_mse_age)

                                # Step the scheduler to change the learning rate
                                if s_mode == "min_loss":
                                    scheduler.step(train_loss)
                                elif s_mode == "min_mse":
                                    scheduler.step(train_loss_age)
                                elif s_mode == "max_acc":
                                    scheduler.step(train_acc_gender)
                                elif s_mode == "max_val_acc":
                                    scheduler.step(val_acc_gender)

                                global_step += 1
                                if train_logger is not None:
                                    train_logger.add_graph(model, img)
                                    train_logger.add_scalar('loss', train_loss, global_step=global_step)
                                    train_logger.add_scalar('acc', train_acc_gender, global_step=global_step)
                                    valid_logger.add_scalar('acc', val_acc_gender, global_step=global_step)
                                    train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                                            global_step=global_step)
                                    if args.age_gender:
                                        train_logger.add_scalar('loss_gender', train_loss_gender,
                                                                global_step=global_step)
                                        train_logger.add_scalar('loss_age', train_loss_age, global_step=global_step)
                                        valid_logger.add_scalar('mse', val_mse_age, global_step=global_step)

                                # Save the model
                                if (epoch % steps_validate == steps_validate - 1) and \
                                        (val_acc_gender > dict_model["acc_gender"] or args.age_gender):
                                    # todo if
                                    print(f"Best val acc (gender, mse) {epoch}: {val_acc_gender}, {val_mse_age}")
                                    name_path = name_model.replace('/', '_')
                                    save_model(model, f"{args.save_path}/{name_path}")
                                    dict_model["acc_gender"] = val_acc_gender
                                    dict_model["mse_age"] = val_mse_age
                                    dict_model["epoch"] = epoch
                                    save_dict(dict_model, f"{args.save_path}/{name_path}.dict")


def test(args) -> None:
    """
    Calculates the metric on the test set of the model given in args.
    Prints the result and saves it in the dictionary files.
    :param args: ArgumentParser with args to run the training (goto main to see the options)
    """
    import pathlib
    device = torch.device('cuda' if torch.cuda.is_available() and not (args.cpu or args.debug) else 'cpu')
    print(device)
    batch_size: int = args.batch_size
    num_workers: int = 0 if args.debug else args.num_workers
    model_names = list(pathlib.Path(args.save_path).glob('*.th'))
    _, _, loader_test = load_data(f"{args.data_path}", num_workers=num_workers,
                                  batch_size=batch_size, lengths=(0.7, 0.15, 0.15))
    model = None
    runs = args.test

    for p in model_names:
        name = p.name.replace('.th', '')
        del model
        model = load_model_from_name(f"{args.save_path}/{name}").to(device)
        loss_age = torch.nn.MSELoss().to(device)

        test_acc_gender = []
        test_mse_age = []
        model.eval()

        for k in range(runs):
            run_acc_gender = []
            run_mse_age = []
            for img, age, gender in loader_test:
                img, age, gender = img.to(device), age.to(device), gender.to(device)
                pred = model(img)

                run_acc_gender.append(accuracy(pred[:, 0], gender))
                if args.age_gender:
                    run_mse_age.append(loss_age(pred[:, 1], age).cpu().detach().numpy())

            test_acc_gender.append(np.mean(run_acc_gender))
            if args.age_gender:
                test_mse_age.append(np.mean(run_mse_age))

        test_acc_gender = np.mean(test_acc_gender)
        dict_result = {"test_acc_gender": test_acc_gender}
        if args.age_gender:
            test_mse_age = np.mean(test_mse_age)
            dict_result["test_mse_age"] = test_mse_age

        print(f"{name}: {dict_result}")
        dict_model.update(dict_result)
        save_dict(dict_model, f"{args.save_path}/{name}.dict")


def predict_age_gender(model: torch.nn.Module, list_imgs: List[str], threshold: float = 0.5,
                       batch_size: int = 32, use_gpu: bool = True) -> torch.Tensor:
    """
    Makes a prediction on the input list of images using a certain model
    :param use_gpu: true to use the gpu
    :param threshold: probability threshold to be considered a woman
    :param batch_size: size of batches to use
    :param model: torch model to use
    :param list_imgs: list of paths of the images used as input of the prediction
    :return: pytorch tensor of predictions over the input images (len(list_images),2)
    """
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(device)

    # batch size cannot higher than length of images
    if len(list_imgs) < batch_size:
        batch_size = len(list_imgs)

    # Load model
    # dict_model = load_dict(f"{model_name}.dict")
    # model = load_model(f"{model_name}.th", CNNClassifier(**dict_model)).to(device)
    model = model.to(device)
    model.eval()

    predictions = []
    for k in range(0, len(list_imgs) - batch_size + 1, batch_size):
        # Load image batch and transform to correct size
        images = list_imgs[k:k + batch_size]
        img_tensor = []
        for p in images:
            img_tensor.append(IMAGE_TRANSFORM(Image.open(p)))
        img_tensor = torch.stack(img_tensor, dim=0).to(device)

        # Predict
        pred = model(img_tensor)  # result (B, 2)
        pred[:, 0] = (torch.sigmoid(pred[:, 0]) > threshold).float()
        predictions.append(pred.cpu().detach())

    return torch.cat(predictions, dim=0)


if __name__ == '__main__':
    # e = predict_age_gender(list_imgs=["C:\\Users\\vibal\\Nextcloud\\Vicente\\Documentos\\Fotos\\Foto 2020.jpg"],
    #                        model_name='./models/savedAgeGender/0.01_adam_h_min_loss_64_[8, 16, 32, 64, 128]_0.1_residual=True_maxPool=True',
    #                        use_gpu=False)

    from argparse import ArgumentParser

    args_parser = ArgumentParser()

    args_parser.add_argument('--log_dir', default="./logs")
    args_parser.add_argument('--data_path', default="./data/UTKFace")
    args_parser.add_argument('--save_path', default="./models/savedAgeGender")
    args_parser.add_argument('--age_gender', action='store_true')
    args_parser.add_argument('-t', '--test', type=int, default=None,
                             help='the number of test runs that will be averaged to give the test result,'
                                  'if None, training mode')

    # Hyper-parameters
    args_parser.add_argument('-lrs', nargs='+', type=float, default=[1e-2], help='learning rates')
    args_parser.add_argument('-law', '--loss_age_weight', nargs='+', type=float, default=[1e-2],
                             help='weight for the age loss')
    # args_parser.add_argument('-opt', '--optimizers', type=str, nargs='+', default=["adam"], help='optimizer to use')
    args_parser.add_argument('-n', '--n_epochs', default=65, type=int, help='number of epochs to train on')
    args_parser.add_argument('-b', '--batch_size', default=64, type=int, help='size of batches to use')
    args_parser.add_argument('-w', '--num_workers', default=2, type=int,
                             help='number of workers to use for data loading')
    args_parser.add_argument('--non_residual', action='store_true',
                             help='if present it will not use residual connections')
    args_parser.add_argument('--non_max_pooling', action='store_true',
                             help='if present the model will not use max pooling (stride in convolutions instead)')
    args_parser.add_argument('--flatten_out_layer', action='store_true',
                             help='if present the model will use flatten before the output linear layer '
                                  'instead of mean pooling')

    args_parser.add_argument('--cpu', action='store_true')
    args_parser.add_argument('-d', '--debug', action='store_true')

    args = args_parser.parse_args()

    if args.test is None:
        train(args)
    else:
        test(args)
