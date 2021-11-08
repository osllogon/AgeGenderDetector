from typing import List, Any, Dict, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pathlib
from glob import glob

LABEL_GENDER = ['man', 'woman']
IMAGE_SIZE = (200, 200)
IMAGE_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGE_SIZE),
    torchvision.transforms.ToTensor()
])


class AgeGenderDataset(Dataset):
    """
    Class that represents a dataset to train the AgeGender classifier
    """

    def __init__(self, image_names: List[pathlib.Path], transform=None):
        """
        Initializer for the dataset
        :param image_names: list of images of the dataset
        :param transform: transformations to be applied to the data when retrieved
        """
        # self.dataset_path = pathlib.Path(dataset_path)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.image_names = image_names

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
        if self.transform is not None:
            image = self.transform(image)

        # image, age, gender
        age, gender = self.get_target(idx)
        return self.to_tensor(image), np.float32(age), np.float32(gender)


def load_data(dataset_path, num_workers=0, batch_size=32, drop_last=False,
              lengths=(0.7, 0.15, 0.15), **kwargs) -> tuple[DataLoader, ...]:
    """
    Method used to load the dataset. It retrives the data with random shuffle
    :param dataset_path: path to the dataset
    :param num_workers: how many subprocesses to use for data loading.
                        0 means that the data will be loaded in the main process
    :param batch_size: size of each batch which is retrieved by the dataloader
    :param drop_last: whether to drop the last batch if it is smaller than batch_size
    :param lengths: tuple with percentage of train, validation and test samples
    :return: tuple of dataloader (same length as parameter lengths)
    """

    # Get list of images and randomly separate them
    image_names = list(pathlib.Path(dataset_path).glob('*.jpg'))
    np.random.default_rng(123).shuffle(image_names)
    lengths = [int(k * len(image_names)) for k in lengths[:-1]]
    lengths = np.cumsum(lengths)

    # Create datasets
    datasets = [AgeGenderDataset(image_names[:lengths[0]], **kwargs)]
    datasets.extend([AgeGenderDataset(image_names[lengths[k]:lengths[k + 1]]) for k in range(len(lengths) - 1)])
    datasets.append(AgeGenderDataset(image_names[lengths[-1]:]))

    # Return DataLoaders for the datasets
    return tuple(DataLoader(k, num_workers=num_workers, batch_size=batch_size, shuffle=True,
                            drop_last=drop_last) for k in datasets)


def accuracy(predicted: torch.Tensor, label: torch.Tensor, mean: bool = True):
    """
    Calculates the accuracy of the prediction and returns a numpy number.
    It considers predicted to be class 1 if probability is higher than 0.5
    :param mean: true to return the mean, false to return an array
    :param predicted: the input prediction
    :param label: the real label
    :return: returns the accuracy of the prediction (between 0 and 1), in the cpu and detached as numpy
    """
    correct = ((predicted > 0).float() == label).float()
    if mean:
        return correct.mean().cpu().detach().numpy()
    else:
        return correct.cpu().detach().numpy()


def save_dict(d: Dict, path: str) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    """
    with open(path, 'w') as file:
        file.write(str(d))


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file in plain text
    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    with open(path, 'r') as file:
        from ast import literal_eval
        loaded = dict(literal_eval(file.read()))
    return loaded


def load_list(path: str) -> List:
    """
    Loads a list from a file in plain text
    :param path: path where the list was saved
    :return: the loaded list
    """
    with open(path, 'r') as file:
        from ast import literal_eval
        loaded = list(literal_eval(file.read()))
    return loaded
