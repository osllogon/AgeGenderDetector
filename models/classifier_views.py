# machine learning imports
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd

# visualization imports
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly
import matplotlib.pyplot as plt

# aux imports
import os
from abc import ABC, abstractmethod

# own imports
from .utils import load_data, accuracy, set_seed
from .models import CNNClassifier

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')


class Visualization(ABC):

    @abstractmethod
    def visualize(self, image: torch.Tensor, gender: int, age: float) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
        pass


class FeatureMaps(Visualization):
    # TODO
    pass


class HeatMap(Visualization):
    """"
    Class for creating and visualizing a heat map by occlusion of different patches from an image

    Attributes
    ----------
    model : str
        A neural net for predicting
    mark_size : int, optional
        The size of the square that it is used for the occlusion of a part of the image. Default value of 16
    scale_factor : int, optional
        The predictions are divided by this factor before applying a softmax so the probabilities do not saturate.
        Default value of 100

    Methods
    -------
    compute_heat_map -> torch.Tensor
        It returns a tensor with the heat map from an image
    visualize_heat_map -> None
        It creates and saves a visualization of a heat map from an image

    """

    def __init__(self, model: torch.nn.Module, mask_size: int = 16, scale_factor: int = 100) -> None:
        self.model = model
        self.mask_size = mask_size
        self.scale_factor = scale_factor

    def compute_heat_maps(self, image: torch.Tensor, gender: int, age: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        It creates a heat map from an image by occlusion technique

        Attributes
        ----------
        image : torch.Tensor
            The image the method receives as input. It has to have 4 dimensions: [1, channels, rows, columns]
        gender: int

        Returns
        -------
        torch.Tensor
            It is a heat map. It has 4 dimensions: [1, channels, rows, columns]
        """

        # Creation of probabilities and average tensor with the same dimensions as the image
        losses_gender = torch.zeros_like(image)
        average_tensor_gender = torch.zeros_like(image)
        losses_age = torch.zeros_like(image)
        average_tensor_age = torch.zeros_like(image)

        loss_gender = torch.nn.BCEWithLogitsLoss().to(device)
        loss_age = torch.nn.MSELoss().to(device)

        # Calculate number of rows and columns
        number_rows = image.size(2)
        number_columns = image.size(3)

        for i in range(number_rows - self.mask_size + 1):
            for j in range(number_columns - self.mask_size + 1):
                # Clone the image so the original it's not modified
                input_ = image.clone()

                # Create the path and compute output
                input_[0, :, i:i + self.mask_size, j:j + self.mask_size] = 0
                output = self.model(input_) / self.scale_factor

                # Actualize probabilities and average tensors
                losses_gender[0, :, i:i + self.mask_size, j:j + self.mask_size] += \
                    loss_gender(output[0, 0], gender)
                average_tensor_gender[0, :, i:i + self.mask_size, j:j + self.mask_size] += 1

                # Actualize probabilities and average tensor
                losses_age[0, :, i:i + self.mask_size, j:j + self.mask_size] += loss_age(output[0, 1], age)
                average_tensor_age[0, :, i:i + self.mask_size, j:j + self.mask_size] += 1

        # Return 1 - probabilities/average due to you want the zone with a higher drop in probability as a hot zone
        return losses_gender / average_tensor_gender, losses_age / average_tensor_age

    # Overriding abstract method
    def visualize(self, image: torch.Tensor, gender: int, age: float) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
        """
        This methods calls compute_heat_map for creating the heat map and then it creates the visualization and saves it

        Attributes
        ----------
        image : torch.Tensor
            This is the image we use as input. It has to have the following dimensions: [1, channels, rows, columns]
        label : str
            The name of the label assigned to this image
        save_path :str
            The image created by the method is saved in this path

        Returns
        -------
        None
            This methods does not return anything

        """

        heat_maps = self.compute_heat_maps(image, gender, age)

        gender_fig = px.imshow(heat_maps[0][0, 0].cpu().detach().numpy())
        gender_fig.update_layout(coloraxis_showscale=False)
        gender_fig.update_xaxes(showticklabels=False)
        gender_fig.update_yaxes(showticklabels=False)

        age_fig = px.imshow(heat_maps[1][0, 0].cpu().detach().numpy())
        age_fig.update_layout(coloraxis_showscale=False)
        age_fig.update_xaxes(showticklabels=False)
        age_fig.update_yaxes(showticklabels=False)

        return gender_fig, age_fig


class SaliencyMap(Visualization):

    def __init__(self, model: torch.nn.Module, threshold: float = 0):
        self.model = model
        self.threshold = threshold

    # Overriding abstract method
    def visualize(self, image: torch.Tensor, gender: int, age: float) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
        """
        This methods generates a saliency map, creates a visualization of it and saves it

        Attributes
        ----------
        image : torch.Tensor
            This is the image we use as input. It has to have the following dimensions: [1, channels, rows, columns]
        label : str
            The name of the label assigned to this image
        save_path :str
            The image created by the method is saved in this path

        Returns
        -------
        None
            This methods does not return anything
        """

        loss_gender = torch.nn.BCEWithLogitsLoss().to(device)
        loss_age = torch.nn.MSELoss().to(device)

        with torch.enable_grad():
            # Pass the image through the model
            input_ = image.clone()
            input_.requires_grad_()
            output = self.model(input_)

            # compute gender loss, clear previous gradients and compute new ones
            loss_val_gender = loss_gender(output[0, 0], gender)
            self.model.zero_grad()
            loss_val_gender.backward()

            # Create and normalize gender saliency map
            gender_saliency_map, _ = torch.max(input_.grad, dim=1)
            gender_saliency_map = (gender_saliency_map - gender_saliency_map.min()) / \
                                  (gender_saliency_map.max() - gender_saliency_map.min())
            gender_saliency_map[gender_saliency_map < self.threshold] = 0

            # pass the input through the model again
            output = self.model(input_)

            # compute age loss, clear previous gradients and compute new ones
            loss_val_age = loss_age(output[0, 1], age)
            self.model.zero_grad()
            input_.grad.zero_()
            loss_val_age.backward()

            # create and normalize age saliency map
            age_saliency_map, _ = torch.max(input_.grad, dim=1)
            age_saliency_map = (age_saliency_map - age_saliency_map.min()) / (
                    age_saliency_map.max() - age_saliency_map.min())
            age_saliency_map[age_saliency_map < self.threshold] = 0

        gender_saliency_map = gender_saliency_map[0].cpu().detach().numpy()
        plt.imshow(gender_saliency_map)
        plt.show()
        gender_saliency_fig = px.imshow(gender_saliency_map)
        gender_saliency_fig.update_layout(coloraxis_showscale=False)
        gender_saliency_fig.update_xaxes(showticklabels=False)
        gender_saliency_fig.update_yaxes(showticklabels=False)

        age_saliency_map = age_saliency_map[0].cpu().detach().numpy()
        age_saliency_fig = px.imshow(age_saliency_map)
        age_saliency_fig.update_layout(coloraxis_showscale=False)
        age_saliency_fig.update_xaxes(showticklabels=False)
        age_saliency_fig.update_yaxes(showticklabels=False)

        return gender_saliency_fig, age_saliency_fig


if __name__ == '__main__':
    print(device)
    print(os.getcwd())
    set_seed(42)

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()])

    loader_train, loader_valid, _ = load_data("./data/UTKFace", num_workers=2,
                                              batch_size=64, transform=transform,
                                              lengths=(0.7, 0.15, 0.15))

    model = CNNClassifier(dim_layers=[32, 64, 128], out_channels=2).to(device)
    model.load_state_dict(torch.load('./models/savedAgeGender/0.01_adam_h_cj9_min_mse_64_[32, 64, '
                                     '128]_0.1_residual=True_maxPool=True.th'))

    object_view = SaliencyMap(model)

    # train_acc_gender = []
    # train_acc_age = []
    with torch.no_grad():
        for img, age, gender in loader_valid:
            # To device
            img, age, gender = img.to(device), age.to(device), gender.to(device)

            # Compute loss and update parameters
            pred = model(img)
            img_index = 30

            original_image = torch.swapaxes(torch.swapaxes(img[img_index], 0, 2), 0, 1)
            original_fig = px.imshow(original_image.cpu().detach().numpy())
            original_fig.update_layout(coloraxis_showscale=False)
            original_fig.update_xaxes(showticklabels=False)
            original_fig.update_yaxes(showticklabels=False)

            clue = gender[img_index]

            gender_figure, age_figure = object_view.visualize(img[img_index].unsqueeze(0),
                                                              gender[img_index], age[img_index])

            original_fig.show()
            gender_figure.show()
            age_figure.show()

            break

    #         train_acc_gender.append(accuracy(pred[:, 0], gender))
    #         train_acc_age.append(accuracy(pred[:, 1], age))
    #
    # print(np.mean(train_acc_gender))
