import itertools
from os import path
from typing import List, Dict

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torchvision.transforms
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.auto import trange, tqdm

from .models import CNNClassifier, save_model, load_model
from .utils import load_data, save_dict, IMAGE_TRANSFORM, ConfusionMatrix


def train(
        model: CNNClassifier,
        dict_model: Dict,
        log_dir: str = "./logs_full",
        data_path: str = "./data/UTKFace",
        save_path: str = "./models/saved_full",
        lr: float = 1e-2,
        optimizer_name: str = "adamw",
        n_epochs: int = 65,
        batch_size: int = 64,
        num_workers: int = 2,
        scheduler_mode: str = 'min_mse',
        debug_mode: bool = False,
        device=None,
        steps_save: int = 1,
        use_cpu: bool = False,
        transforms: List = [torchvision.transforms.RandomHorizontalFlip()],
        loss_age_weight: float = 1e-2,
):
    """
    Method that trains a given model

    :param model: model that will be trained
    :param dict_model: dictionary of model parameters
    :param log_dir: directory where the tensorboard log should be saved
    :param data_path: directory where the data can be found
    :param save_path: directory where the model will be saved
    :param lr: learning rate for the training
    :param optimizer_name: optimizer used for training. Can be `adam, adamw, sgd`
    :param n_epochs: number of epochs of training
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param scheduler_mode: scheduler mode to use for the learning rate scheduler. Can be `min_loss, min_mse, max_acc, max_val_acc, max_val_mcc`
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param device: if not none, device to use ignoring other parameters. If none, the device will be used depending on `use_cpu` and `debug_mode` parameters
    :param steps_save: number of epoch after which to validate and save model (if conditions met)
    :param transforms: transformations to apply to the training data for data augmentation
    :param loss_age_weight: weight for the age loss
    """

    # cpu or gpu used for training if available (gpu much faster)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    # print(device)

    # num_workers 0 if debug_mode
    if debug_mode:
        num_workers = 0

    # Tensorboard
    global_step = 0
    # dictionary of training parameters
    dict_param = {f"train_{k}": v for k, v in locals().items() if k in [
        'lr',
        'optimizer_name',
        'batch_size',
        'scheduler_mode',
        'transforms',
        'loss_age_weight',
    ]}
    # dictionary to set model name
    name_dict = dict_model.copy()
    name_dict.update(dict_param)
    # model name
    name_model = '/'.join([
        str(name_dict)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
    ])

    # train_logger = tb.SummaryWriter(path.join(log_dir, 'train', name_model), flush_secs=1)
    # valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid', name_model), flush_secs=1)
    train_logger = tb.SummaryWriter(path.join(log_dir, name_model), flush_secs=1)
    valid_logger = train_logger

    # Model
    dict_model.update(dict_param)
    # dict_model.update(dict(
    #     # metrics
    #     train_loss = None,
    #     train_loss = None,
    #     train_acc = None,
    #     val_acc = None,
    #     val_mcc = None,
    #     epoch=0,
    # ))
    model = model.to(device)

    # Loss
    loss_gender = torch.nn.BCEWithLogitsLoss().to(device)  # sigmoid + BCELoss (good for 2 classes classification)
    loss_age = torch.nn.MSELoss().to(device)

    # load train and test data
    # todo random seed 123
    loader_train, loader_valid, _ = load_data(
        dataset_path=data_path,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        random_seed=4444,
        transform=transforms,
    )

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not configured")

    # :param scheduler_patience: value used as patience for the learning rate scheduler
    if scheduler_mode in ["min_loss", 'min_mse']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    elif scheduler_mode in ["max_acc", "max_val_acc", "max_val_mcc"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    else:
        raise Exception("Optimizer not configured")

    met = None
    # print(f"{log_dir}/{name_model}")
    for epoch in (p_bar := trange(n_epochs)):
        p_bar.set_description(f"{name_model} -> {met if met is not None else ''}")
        # print(epoch)
        train_loss = []
        train_loss_gender = []
        train_loss_age = []
        train_cm = ConfusionMatrix(size=2, name='train')

        # Start training: train mode and freeze bert
        model.train()
        for img, age, gender in loader_train:
            # To device
            img, age, gender = img.to(device), age.to(device), gender.to(device)

            # Compute loss and update parameters
            pred = model(img)
            loss_val_gender = loss_gender(pred[:, 0], gender)
            loss_val_age = loss_age(pred[:, 1], age)
            loss_val = loss_val_gender + loss_age_weight * loss_val_age

            # Do back propagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Add train loss and accuracy
            train_loss.append(loss_val.cpu().detach().numpy())
            train_loss_age.append(loss_val_age.cpu().detach().numpy())
            train_loss_gender.append(loss_val_gender.cpu().detach().numpy())
            train_cm.add(preds=(pred[:, 0] > 0).float(), labels=gender)

        # Evaluate the model
        val_cm = ConfusionMatrix(size=2, name='val')
        val_mse_age = []
        model.eval()
        with torch.no_grad():
            for img, age, gender in loader_valid:
                # To device
                img, age, gender = img.to(device), age.to(device), gender.to(device)
                pred = model(img)

                val_cm.add((pred[:, 0] > 0).float(), gender)
                val_mse_age.append(loss_age(pred[:, 1], age).cpu().detach().numpy())

        # mean loss values
        train_loss = np.mean(train_loss)
        train_loss_gender = np.mean(train_loss_gender)
        train_loss_age = np.mean(train_loss_age)
        # mse
        val_mse_age = np.mean(val_mse_age)

        # Step the scheduler to change the learning rate
        is_better = False
        if scheduler_mode == "min_loss":
            met = train_loss
            if (best_met := dict_model.get('train_loss', None)) is not None:
                is_better = met <= best_met
        elif scheduler_mode == "min_mse":
            met = val_mse_age
            if (best_met := dict_model.get('val_mse_age', None)) is not None:
                is_better = met <= best_met
        elif scheduler_mode == "max_acc":
            met = train_cm.global_accuracy
            if (best_met := dict_model.get('train_acc', None)) is not None:
                is_better = met >= best_met
        elif scheduler_mode == "max_val_acc":
            met = val_cm.global_accuracy
            if (best_met := dict_model.get('val_acc', None)) is not None:
                is_better = met >= best_met
        elif scheduler_mode == 'max_val_mcc':
            met = val_cm.matthews_corrcoef
            if (best_met := dict_model.get('val_mcc', None)) is not None:
                is_better = met >= best_met
        else:
            met = None

        if met is not None:
            scheduler.step(met)

        global_step += 1
        if train_logger is not None:
            suffix = 'train'
            train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)
            train_logger.add_scalar(f'loss_age_{suffix}', train_loss_age, global_step=global_step)
            train_logger.add_scalar(f'loss_gender_{suffix}', train_loss_gender, global_step=global_step)
            log_confussion_matrix(train_logger, train_cm, global_step, suffix=suffix)
            # validation log
            suffix = 'val'
            valid_logger.add_scalar(f'mse_age_{suffix}', val_mse_age, global_step=global_step)
            log_confussion_matrix(valid_logger, val_cm, global_step, suffix=suffix)
            # lr
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # Save the model
        if (epoch % steps_save == steps_save - 1) or is_better:
            d = dict_model if is_better else dict_model.copy()

            # print(f"Best val acc {epoch}: {val_acc}")
            d["epoch"] = epoch + 1
            # metrics
            d["train_loss"] = train_loss
            d["val_mse_age"] = val_mse_age
            d["train_acc"] = train_cm.global_accuracy
            d["val_acc"] = val_cm.global_accuracy
            d["val_mcc"] = val_cm.matthews_corrcoef

            name_path = str(list(name_dict.values()))[1:-1].replace(',', '_').replace("'", '').replace(' ', '')
            # name_path = f"{d['val_acc']:.2f}_{name_path}"
            # if periodic save, then include epoch
            if not is_better:
                name_path = f"{name_path}_{epoch + 1}"
            save_model(model, save_path, name_path, param_dicts=d)


def log_confussion_matrix(logger, confussion_matrix: ConfusionMatrix, global_step: int, suffix=''):
    """
    Logs the data in the confussion matrix to a logger
    :param logger: tensorboard logger to use for logging
    :param confussion_matrix: confussion matrix from where the metrics are obtained
    :param global_step: global step for the logger
    """
    logger.add_scalar(f'acc_global_{suffix}', confussion_matrix.global_accuracy, global_step=global_step)
    logger.add_scalar(f'acc_avg_{suffix}', confussion_matrix.average_accuracy, global_step=global_step)
    logger.add_scalar(f'mcc_{suffix}', confussion_matrix.matthews_corrcoef, global_step=global_step)
    for idx, k in enumerate(confussion_matrix.class_accuracy):
        logger.add_scalar(f'acc_class_{idx}_{suffix}', k, global_step=global_step)


def test(
        data_path: str = "./data/UTKFace",
        save_path: str = './models/saved_full',
        n_runs: int = 1,
        batch_size: int = 8,
        num_workers: int = 0,
        debug_mode: bool = False,
        use_cpu: bool = False,
        save: bool = True,
        verbose: bool = False,
) -> None:
    """
    Calculates the metric on the test set of the model given in args.
    Prints the result and saves it in the dictionary files.

    :param data_path: directory where the data can be found
    :param save_path: directory where the model will be saved
    :param n_runs: number of runs from which to take the mean
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param save: whether to save the results in the model dict
    :param verbose: whether to print results
    """

    def print_v(s):
        if verbose:
            print(s)

    from pathlib import Path
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print_v(device)
    # num_workers 0 if debug_mode
    if debug_mode:
        num_workers = 0

    # get model names from folder
    model = None
    # best_dict = None
    # best_acc = 0.0
    list_all = []
    paths = list(Path(save_path).glob('*'))
    for folder_path in tqdm(paths):
        print_v(f"Testing {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()
        loader_train, loader_valid, loader_test = load_data(
            dataset_path=data_path,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=False,
            random_seed=4444,
            transform=transforms,
        )

        # start testing
        train_mse = []
        train_mae = []
        train_cm = []
        val_mse = []
        val_mae = []
        val_cm = []
        test_mse = []
        test_mae = []
        test_cm = []
        for k in range(n_runs):
            train_run_mse = []
            train_run_mae = []
            train_run_cm = ConfusionMatrix(size=2, name='train')
            val_run_mse = []
            val_run_mae = []
            val_run_cm = ConfusionMatrix(size=2, name='val')
            test_run_mse = []
            test_run_mae = []
            test_run_cm = ConfusionMatrix(size=2, name='test')

            with torch.no_grad():
                # train
                for img, age, gender in loader_train:
                    img, age, gender = img.to(device), age.to(device), gender.to(device)
                    pred = model(img)

                    train_run_mse.append(mean_squared_error(y_true=age, y_pred=pred[:, 1]).cpu().detach().numpy())
                    train_run_mae.append(mean_absolute_error(y_true=age, y_pred=pred[:, 1]).cpu().detach().numpy())
                    train_run_cm.add(preds=(pred[:, 0] > 0).float(), labels=gender)

                # valid
                for img, age, gender in loader_valid:
                    img, age, gender = img.to(device), age.to(device), gender.to(device)
                    pred = model(img)

                    val_run_mse.append(mean_squared_error(y_true=age, y_pred=pred[:, 1]).cpu().detach().numpy())
                    val_run_mae.append(mean_absolute_error(y_true=age, y_pred=pred[:, 1]).cpu().detach().numpy())
                    val_run_cm.add(preds=(pred[:, 0] > 0).float(), labels=gender)

                # test
                for img, age, gender in loader_test:
                    img, age, gender = img.to(device), age.to(device), gender.to(device)
                    pred = model(img)

                    test_run_mse.append(mean_squared_error(y_true=age, y_pred=pred[:, 1]).cpu().detach().numpy())
                    test_run_mae.append(mean_absolute_error(y_true=age, y_pred=pred[:, 1]).cpu().detach().numpy())
                    test_run_cm.add(preds=(pred[:, 0] > 0).float(), labels=gender)

            print_v(f"Run {k}: {test_run_cm.global_accuracy}")

            train_mse.append(np.mean(train_run_mse))
            train_mae.append(np.mean(train_run_mae))
            train_cm.append(train_run_cm)
            val_mse.append(np.mean(val_run_mse))
            val_mae.append(np.mean(val_run_mae))
            val_cm.append(val_run_cm)
            test_mse.append(np.mean(test_run_mse))
            test_mae.append(np.mean(test_run_mae))
            test_cm.append(test_run_cm)

        dict_result = {
            "train_mae": np.mean(train_mae),
            "val_mae": np.mean(val_mae),
            "test_mae": np.mean(test_mae),

            "train_mse": np.mean(train_mse),
            "val_mse": np.mean(val_mse),
            "test_mse": np.mean(test_mse),

            "train_mcc": np.mean([k.matthews_corrcoef for k in train_cm]),
            "val_mcc": np.mean([k.matthews_corrcoef for k in val_cm]),
            "test_mcc": np.mean([k.matthews_corrcoef for k in test_cm]),

            "train_acc": np.mean([k.global_accuracy for k in train_cm]),
            "val_acc": np.mean([k.global_accuracy for k in val_cm]),
            "test_acc": np.mean([k.global_accuracy for k in test_cm]),
        }

        print_v(f"RESULT: {dict_result}")

        dict_model.update(dict_result)
        if save:
            save_dict(dict_model, f"{folder_path}/{folder_path.name}.dict")

        list_all.append(dict(
            dict=dict_model,
            train_cm=train_cm,
            val_cm=val_cm,
            test_cm=test_cm,
        ))

        # # save if best
        # if best_acc < (test_acc := dict_model['test_acc']):
        #     best_acc = test_acc
        #     best_dict = dict_model

    return list_all


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
    dict_model = dict(
        # dictionary with model information
        in_channels=[3],
        out_channels=[2],
        dim_layers=[[32, 64, 128]],
        block_conv_layers=[3],
        residual=[True],
        max_pooling=[True, False],
        # training param
        transforms=[torchvision.transforms.RandomHorizontalFlip()]
    )

    list_model = [dict(zip(dict_model.keys(), k)) for k in itertools.product(*dict_model.values())]

    for d in list_model:
        d = d.copy()
        transforms = d.pop('transforms')

        train(
            model=CNNClassifier(**d),
            dict_model=d,
            log_dir="./logs_full",
            data_path="./data/UTKFace",
            save_path="./models/saved_full",
            lr=1e-2,
            optimizer_name="adamw",
            n_epochs=65,
            batch_size=64,
            num_workers=2,
            scheduler_mode='min_mse',
            debug_mode=False,
            device=None,
            steps_save=1,
            use_cpu=False,
            transforms=transforms,
            loss_age_weight=1e-2,
        )

#     from argparse import ArgumentParser
#     args_parser = ArgumentParser()

#     args_parser.add_argument('-t', '--test', type=int, default=None,
#                              help='the number of test runs that will be averaged to give the test result,'
#                                   'if None, training mode')

#     args = args_parser.parse_args()

#     if args.test is not None:
#         test(
#             n_runs=args.test,
#             save_path="./models/saved_full"
#         )
#     else:
#         # Model
#         dict_model = {
#             # dictionary with model information
#             "in_channels": [3],
#             "out_channels": [2 if args.age_gender else 1],
#             "dim_layers": [[32, 64, 128]],
#             "block_conv_layers": [3],
#             "residual": [True],
#             "max_pooling": [True, False],
#         }
#         model = CNNClassifier(**dict_model)

#         train(
#             model,
#             dict_model,
#             log_dir = "./logs_full",
#             data_path = "./data/UTKFace",
#             save_path = "./models/saved_full",
#             lr = 1e-2,
#             optimizer_name = "adamw",
#             n_epochs = 65,
#             batch_size = 64,
#             num_workers = 2,
#             scheduler_mode = 'min_mse',
#             debug_mode = False,
#             device = None,
#             steps_save = 1,
#             use_cpu = False,
#             transforms = [torchvision.transforms.RandomHorizontalFlip()],
#             loss_age_weight = 1e-2,
#         )
