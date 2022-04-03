import os
import pickle
from random import random
from typing import List, Any, Dict, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pathlib
import pandas as pd
import seaborn as sns
from glob import glob

LABEL_GENDER = ['man', 'woman']
IMAGE_SIZE_CROP = (200, 200)
IMAGE_TRANSFORM_CROP = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGE_SIZE_CROP),
    torchvision.transforms.ToTensor()
])


class AgeGenderDataset(Dataset):
    """
    Class that represents a dataset to train the AgeGender classifier
    """

    def __init__(self, image_names: List[pathlib.Path], transform=None, raw:bool = False):
        """
        Initializer for the dataset
        :param image_names: list of images of the dataset
        :param transform: transformations to be applied to the data when retrieved
        :param raw: whether to give the images in a non-tensor format
        """
        # self.dataset_path = pathlib.Path(dataset_path)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.image_names = image_names
        self.raw = raw

    def __len__(self):
        """
        :return: the length of the dataset (how many images it has)
        """
        return len(self.image_names)

    def get_target(self, idx) -> Tuple[float, float]:
        """
        Returns the target values from the dataset given the index
        :param idx: index to retrieve
        :return: target values retrieved (age: float, gender: float)
        """
        # images: edad_género_raza_datosirrelevantes.jpg.chip.jpg
        name_split = self.image_names[idx].name.split('_')
        # age, gender
        return float(name_split[0]), float(name_split[1])

    def __getitem__(self, idx):
        """
        Returns an item from the dataset given the index
        :param idx: index to retrieve
        :return: item retrieved (image: torch.Tensor, age: float, gender: float)
        """
        image = Image.open(self.image_names[idx])
        # Apply transformation to image
        if not self.raw:
            if self.transform is not None:
                image = self.transform(image)
            image = self.to_tensor(image)

        # image, age, gender
        age, gender = self.get_target(idx)
        return image, np.float32(age), np.float32(gender)


def load_data(dataset_path, num_workers=0, batch_size=32, drop_last=False,
              lengths=(0.7, 0.15, 0.15), random_seed=4444, raw_val:bool = False, **kwargs) -> tuple[DataLoader, ...]:
    """
    Method used to load the dataset. It retrives the data with random shuffle
    :param dataset_path: path to the dataset
    :param num_workers: how many subprocesses to use for data loading.
                        0 means that the data will be loaded in the main process
    :param batch_size: size of each batch which is retrieved by the dataloader
    :param drop_last: whether to drop the last batch if it is smaller than batch_size
    :param lengths: tuple with percentage of train, validation and test samples
    :param seed: seed for the loaders
    :param raw_pred: whether to give validation data in a non-tensor format
    :return: tuple of dataloader (same length as parameter lengths)
    """

    # Get list of images and randomly separate them
    image_names = list(pathlib.Path(dataset_path).glob('*.jpg'))
    np.random.default_rng(random_seed).shuffle(image_names)
    lengths = [int(k * len(image_names)) for k in lengths[:-1]]
    lengths = np.cumsum(lengths)

    # Create datasets
    datasets = [AgeGenderDataset(image_names[:lengths[0]], **kwargs)]
    datasets.extend([AgeGenderDataset(image_names[lengths[k]:lengths[k + 1]], raw=raw_val) for k in range(len(lengths) - 1)])
    datasets.append(AgeGenderDataset(image_names[lengths[-1]:], raw=raw_val))

    # Return DataLoaders for the datasets
    return tuple(DataLoader(k, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                            drop_last=drop_last and idx!=len(datasets) - 1) for idx,k in enumerate(datasets))


class ConfusionMatrix:
    """
    Class that represents a confusion matrix.

    Cij is equal to the number of observations known to be in class i and predicted in class j
    """

    def _make(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Returns the confusion matrix of the given predicted and labels values
        :param preds: predicted values (B)
        :param labels: true values (B)
        :return: (size,size) confusion matrix of `size` classes
        """
        matrix = torch.zeros(self.size, self.size, dtype=torch.float)
        for t, p in zip(labels.reshape(-1).long(), preds.reshape(-1).long()):
            matrix[t, p] += 1
        return matrix

    def __init__(self, size=5, name: str = ''):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        :param name: name of the confusion matrix
        """
        self.matrix = torch.zeros(size, size, dtype=torch.float)
        self.preds = None
        self.labels = None
        self.name = name

    def __repr__(self) -> str:
        return self.matrix.numpy().__repr__

    def add(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        :param preds: predicted values (B)
        :param labels: true values (B)
        """
        preds = preds.reshape(-1).cpu().detach().clone()
        labels = labels.reshape(-1).cpu().detach().clone()

        self.matrix += self._make(preds, labels)
        self.preds = torch.cat((self.preds, preds), dim=0) if self.preds is not None else preds
        self.labels = torch.cat((self.labels, labels), dim=0) if self.labels is not None else labels

    @property
    def size(self):
        return self.matrix.shape[0]

    @property
    def matthews_corrcoef(self):
        """Matthews correlation coefficient (MCC)"""
        return matthews_corrcoef(y_true=self.labels.numpy(), y_pred=self.preds.numpy())

    @property
    def rmse(self):
        return mean_squared_error(y_true=self.labels.numpy(), y_pred=self.preds.numpy(), squared=False)

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean().item()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

    @property
    def normalize(self):
        return self.matrix / (self.matrix.sum() + 1e-5)

    def visualize(self, normalize: bool = False):
        """
        Visualize confusion matrix
        :param normalize: whether to normalize the matrix by the total amount of samples
        """
        plt.figure(figsize=(15, 10))

        matrix = self.normalize.numpy() if normalize else self.matrix.numpy()

        df_cm = pd.DataFrame(matrix).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return plt


def save_dict(d: Dict, path: str, as_str: bool = False) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    :param as_str: If true, it will save as a string. If false, it will use pickle
    """
    if as_str:
        with open(path, 'w', encoding="utf-8") as file:
            file.write(str(d))
    else:
        with open(path, 'wb') as file:
            pickle.dump(d, file)


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file (plain text or pickle)
    :param path: path where the dictionary was saved

    :return: the loaded dictionary
    """
    with open(path, 'rb') as file:
        try:
            return pickle.load(file)
        except pickle.UnpicklingError as e:
            # print(e)
            pass

    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        s = file.read()
        return dict(literal_eval(s))


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    :param seed: seed for the random generators
    """
    # todo delete all calls to set seed except this one
    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        # for deterministic behavior on cuda >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def save_pickle(obj, save_path:str):
    """
    Saves an object with pickle
    :param obj: object to be saved
    :param save_path: path to the file where it will be saved
    """
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path:str):
    """
    Loads an object with pickle from a file
    :param path: path to the file where the object is stored
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    data = load_data(
        dataset_path='./data_full',
        num_workers=0,
        batch_size=1,
        drop_last=False,
        random_seed=4444,
    )
    l = [k[0].shape for k in data[0]]
    print('Done')