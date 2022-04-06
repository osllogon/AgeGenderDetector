from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas
import plotly
import plotly.graph_objects as go
import plotly.express as px
import torch
import torchvision
from pandas import DataFrame
from plotly.subplots import make_subplots

from models.models import CNNClassifier, load_model
from models.utils import load_data, LABEL_GENDER, load_dict, accuracy, set_seed

DATASET_COLORS = {"Train": "green", "Valid": "blue", "Test": "red"}
GENRE_COLORS = dict(zip(LABEL_GENDER, ['skyblue', 'pink']))


def get_datasets(data_path: str = "./data/UTKFace") -> List[DataFrame]:
    """
    Get train, validation, test datasets
    :param data_path: path to the data
    :return: list of datasets (train, validation, test)
    """
    # Get data
    datasets = [DataFrame([ds.dataset.get_target(k) for k in range(len(ds.dataset))],
                          columns=["Age", "Gender"], index=ds.dataset.image_names) for ds in load_data(data_path)]
    return datasets


def visualize_datasets_age(datasets: List[DataFrame]) -> Tuple[go.Figure, go.Figure]:
    """
    Visualization for the age distributions in datasets used for training
    :param datasets: list of datasets obtained from the get_datasets method
    :return: tuple of plotly figures (2)
    """
    max_age = max([k["Age"].max() for k in datasets])
    min_age = min([k["Age"].min() for k in datasets])

    # Age distribution of datasets
    fig_bar = go.Figure()
    for data, name in zip(datasets, DATASET_COLORS.keys()):
        fig_bar.add_trace(
            go.Histogram(
                x=data["Age"],
                marker_color=DATASET_COLORS[name],
                xbins=dict(
                    start=min_age,
                    end=max_age,
                    size=5
                ),
                histnorm="probability",
                opacity=0.7,
                name=name,
                showlegend=True
            )
        )
    fig_bar.update_layout(title="Age distribution of sets", xaxis_title="Age", yaxis_title="Probability (%)")

    fig_box = go.Figure()
    for data, name in zip(datasets, DATASET_COLORS.keys()):
        fig_box.add_trace(
            go.Box(
                y=data["Age"],
                marker_color=DATASET_COLORS[name],
                name=name,
                boxmean=True,
                showlegend=True
            )
        )
    fig_box.update_layout(title="Age distribution of sets", yaxis_title="Age")

    return fig_bar, fig_box


def visualize_datasets_gender(datasets) -> go.Figure:
    """
    Visualization for the gender distributions in datasets used for training
    :param datasets: list of datasets obtained from the get_datasets method
    :return: plotly figure
    """
    # Map values
    datasets = [k["Gender"].map({0.0: LABEL_GENDER[0], 1.0: LABEL_GENDER[1]}).value_counts(normalize=True)
                for k in datasets]

    # Gender distribution of datasets
    fig_bar = go.Figure()
    for data, name in zip(datasets, DATASET_COLORS.keys()):
        fig_bar.add_trace(
            go.Bar(
                x=data.index,
                y=data,
                marker_color=DATASET_COLORS[name],
                opacity=0.7,
                name=name,
                showlegend=True
            )
        )
    fig_bar.update_layout(title="Gender distribution of sets", xaxis_title="Gender", yaxis_title="Probability (%)")

    return fig_bar


# def visualize_data_images(datasets):
#     # Show multiple random photos
#     number_images = 4
#     fig = go.Figure()
#     for k in range(number_images):
#         # pytorch tensor to numpy

#         fig.add_trace(
#             go.Image(
#                 # z=
#             )
#         )


def visualize_model(model: torch.nn.Module, data_path, num_workers: int = 0, batch_size: int = 64,
                    use_gpu: bool = True) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Visualize accuracy and RMSE per dataset and per age
    :param model: torch model to use
    :param data_path: path to the data images
    :param num_workers: number of workers (processes) to use for data loading
    :param batch_size: size of batches to use
    :param use_gpu: whether to use the gpu (true) or cpu (false)
    :return: tuple of plotly figures
    """
    loaders = load_data(data_path, num_workers=num_workers, batch_size=batch_size, transform=None)
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # Load model and loss
    # model = load_model(f"{model_name}.th", CNNClassifier(**(load_dict(f"{model_name}.dict")))).to(device)
    model = model.to(device)
    loss_age = torch.nn.MSELoss(reduction='none').to(device)

    dfs = []
    for dataset_name, loader in zip(DATASET_COLORS.keys(), loaders):
        ages = []
        genders = []
        gender_acc = []
        age_se = []
        for img, age, gender in loader:
            img, age, gender = img.to(device), age.to(device), gender.to(device)
            pred = model(img)

            ages.extend(age.cpu().detach().numpy())
            genders.extend(gender.cpu().detach().numpy())
            gender_acc.extend(accuracy(pred[:, 0], gender, mean=False))
            age_se.extend(loss_age(pred[:, 1], age).cpu().detach().numpy())

        df = DataFrame({"age": ages, "gender": genders, "gender_acc": gender_acc, "age_se": age_se})
        df['dataset'] = dataset_name
        dfs.append(df)

    # del model
    df = pandas.concat(dfs, axis=0, ignore_index=True)

    # Visualize train vs validation vs test metrics
    df["gender"] = df["gender"].map({0.0: LABEL_GENDER[0], 1.0: LABEL_GENDER[1]})
    dataset_metrics = df[["dataset", "gender_acc", "age_se"]].groupby("dataset").mean()
    dataset_metrics['age_se'] = np.sqrt(dataset_metrics['age_se'])

    fig_dataset = make_subplots(rows=1, cols=2, subplot_titles=("Gender accuracy per dataset", "Age RMSE per dataset"))
    fig_dataset.add_trace(
        go.Bar(
            x=dataset_metrics['gender_acc'] * 100,
            y=dataset_metrics.index,
            orientation='h',
            marker_color=list(DATASET_COLORS.values()),
            showlegend=False
        ),
        row=1,
        col=1
    )
    fig_dataset.update_yaxes(title_text="Datasets", row=1, col=1)
    fig_dataset.update_xaxes(title_text="Accuracy (%)", row=1, col=1)

    fig_dataset.add_trace(
        go.Bar(
            x=dataset_metrics['age_se'],
            y=dataset_metrics.index,
            orientation='h',
            marker_color=list(DATASET_COLORS.values()),
            showlegend=False
        ),
        row=1,
        col=2
    )
    fig_dataset.update_yaxes(title_text="Datasets", row=1, col=2)
    fig_dataset.update_xaxes(title_text="RMSE", row=1, col=2)

    # Visualize age RMSE by age
    df_no_train = df[df['dataset'] != list(DATASET_COLORS.keys())[0]]
    df_no_train.sort_values(by=['age'], ascending=True, inplace=True)

    df_ages_rmse = df_no_train[['age', 'gender', 'age_se']]
    df_gender_acc_all = df_ages_rmse.groupby('age').mean()
    df_ages_rmse_single = [df_ages_rmse[df_ages_rmse['gender'] == label].groupby('age').mean() for label in
                           LABEL_GENDER]

    fig_age = make_subplots(rows=1, cols=2, subplot_titles=("Age RMSE per age (valid and test)",
                                                            "Gender accuracy per age (valid and test)"))

    fig_age.add_trace(
        go.Scatter(
            x=df_gender_acc_all.index,
            y=np.sqrt(df_gender_acc_all['age_se']),
            mode="lines",
            line=dict(color='gray', dash='dash'),
            name='both',
            showlegend=True
        ),
        row=1,
        col=1
    )
    fig_age.update_yaxes(title_text="RMSE", row=1, col=1)
    fig_age.update_xaxes(title_text="Age", row=1, col=1)

    for name, df in zip(LABEL_GENDER, df_ages_rmse_single):
        fig_age.add_trace(
            go.Scatter(
                x=df.index,
                y=np.sqrt(df['age_se']),
                mode="lines",
                line=dict(color=GENRE_COLORS[name]),
                name=name,
                showlegend=True
            ),
            row=1,
            col=1
        )

    # Visualize gender accuracy by age
    df_gender_acc = df_no_train[['age', 'gender', 'gender_acc']]
    max_age = df_gender_acc['age'].max()
    min_age = df_gender_acc['age'].min()

    fig_age.add_trace(
        go.Histogram(
            x=df_gender_acc['age'],
            y=df_gender_acc['gender_acc'],
            marker_color='gray',
            xbins=dict(
                start=min_age,
                end=max_age,
                size=10
            ),
            histfunc='avg',
            name='both',
            showlegend=True
        ),
        row=1,
        col=2
    )
    for name in LABEL_GENDER:
        df = (df_gender_acc[df_gender_acc['gender'] == name])
        fig_age.add_trace(
            go.Histogram(
                x=df['age'],
                y=df['gender_acc'],
                marker_color=GENRE_COLORS[name],
                xbins=dict(
                    start=min_age,
                    end=max_age,
                    size=10
                ),
                histfunc='avg',
                name=name,
                showlegend=True
            ),
            row=1,
            col=2
        )
    fig_age.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig_age.update_xaxes(title_text="Age", row=1, col=2)

    # Same but with a line
    df_gender_acc_all = df_gender_acc.groupby('age').mean()
    df_gender_acc_single = [df_gender_acc[df_gender_acc['gender'] == label].groupby('age').mean() for label in
                            LABEL_GENDER]

    fig_age_line = go.Figure()
    fig_age_line.add_trace(
        go.Scatter(
            x=df_gender_acc_all.index,
            y=df_gender_acc_all['gender_acc'] * 100,
            mode="lines",
            line=dict(color='gray', dash='dash'),
            name='both',
            showlegend=True
        )
    )
    for name, df in zip(LABEL_GENDER, df_gender_acc_single):
        fig_age_line.add_trace(
            go.Scatter(
                x=df.index,
                y=df['gender_acc'] * 100,
                mode="lines",
                line=dict(color=GENRE_COLORS[name]),
                name=name,
                showlegend=True
            )
        )
    fig_age_line.update_layout(title="Gender accuracy per age (valid and test)", xaxis_title="Age",
                               yaxis_title="Accuracy (%)")

    return fig_dataset, fig_age, fig_age_line


def visualize_training_data(data_path: str) -> plotly.graph_objs.Figure:
    # TODO
    pass


class Visualization(ABC):
    @abstractmethod
    def visualize(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
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

    def compute_heat_maps(self, image: torch.Tensor, gender: int, age: float, device: torch.device) \
            -> tuple[torch.Tensor, torch.Tensor]:
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
    def visualize(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
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

        # create device and print it
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(device)

        # pass everything to device
        image = image.to(device)
        self.model = self.model.to(device)

        # convert age and gender into tensors
        age = torch.Tensor([age])[0]
        gender = torch.Tensor([gender])[0]

        heat_maps = self.compute_heat_maps(image, gender, age, device)

        gender_fig = px.imshow(heat_maps[0][0, 0].cpu().detach().numpy())
        gender_fig.update_layout(coloraxis_showscale=False)
        gender_fig.update_xaxes(showticklabels=False)
        gender_fig.update_yaxes(showticklabels=False)

        age_fig = px.imshow(heat_maps[1][0, 0].cpu().detach().numpy())
        age_fig.update_layout(coloraxis_showscale=False)
        age_fig.update_xaxes(showticklabels=False)
        age_fig.update_yaxes(showticklabels=False)

        return gender_fig, age_fig


class SaliencyMap:
    def __init__(self, model: torch.nn.Module, threshold: float):
        self.model = model
        self.threshold = threshold

    def compute_saliency_map(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool) -> tuple[
        torch.Tensor, torch.Tensor]:
        # create device and print it
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(device)

        # pass everything to device
        image = image.to(device)
        self.model = self.model.to(device)

        # convert age and gender into tensors
        age = torch.Tensor([age])[0].to(device)
        gender = torch.Tensor([gender])[0].to(device)

        loss_gender = torch.nn.BCEWithLogitsLoss().to(device)
        loss_age = torch.nn.MSELoss().to(device)

        with torch.enable_grad():
            # pass the image through the model
            input_ = image.clone()
            input_.requires_grad_()
            output = self.model(input_)

            # compute gender loss, clear previous gradients and compute new ones
            loss_val_gender = loss_gender(output[0, 0], gender)
            self.model.zero_grad()
            loss_val_gender.backward()

            # create and normalize gender saliency map
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

            return gender_saliency_map, age_saliency_map


class OriginalSaliencyMap(Visualization, SaliencyMap):

    def __init__(self, model: torch.nn.Module, threshold: float = 0):
        super().__init__(model, threshold)

    # Overriding abstract method
    def visualize(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
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

        gender_saliency_map, age_saliency_map = self.compute_saliency_map(image, gender, age, use_gpu)

        gender_saliency_map = gender_saliency_map[0].cpu().detach().numpy()
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


class SaliencyMapCombined(Visualization, SaliencyMap):

    def __init__(self, model: torch.nn.Module, threshold: float = 0):
        super().__init__(model, threshold)

    # Overriding abstract method
    def visualize(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
        """
        This methods generates a saliency map combined with the original image, creates a visualization of it and saves it

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

        gender_saliency_map, age_saliency_map = self.compute_saliency_map(image, gender, age, use_gpu)

        saliency_map_combined_age = image[0] * age_saliency_map
        saliency_map_combined_age = torch.swapaxes(torch.swapaxes(saliency_map_combined_age, 0, 2), 0, 1)
        saliency_map_combined_age = px.imshow(saliency_map_combined_age.detach().numpy())
        saliency_map_combined_age.update_layout(coloraxis_showscale=False)
        saliency_map_combined_age.update_xaxes(showticklabels=False)
        saliency_map_combined_age.update_yaxes(showticklabels=False)

        saliency_map_combined_gender = image[0] * gender_saliency_map
        saliency_map_combined_gender = torch.swapaxes(torch.swapaxes(saliency_map_combined_gender, 0, 2), 0, 1)
        saliency_map_combined_gender = px.imshow(saliency_map_combined_gender.detach().numpy())
        saliency_map_combined_gender.update_layout(coloraxis_showscale=False)
        saliency_map_combined_gender.update_xaxes(showticklabels=False)
        saliency_map_combined_gender.update_yaxes(showticklabels=False)

        return saliency_map_combined_gender, saliency_map_combined_age


class IntegratedSaliencyMap(Visualization, SaliencyMap):

    def __init__(self, model: torch.nn.Module, threshold: float = 0):
        super().__init__(model, threshold)

    def _mean_saliency_maps(self, saliency_map: torch.Tensor) -> torch.Tensor:
        height = saliency_map.size(1)
        width = saliency_map.size(0)

        horizontal_tensor = torch.zeros(1, height)
        vertical_tensor = torch.zeros(width, 1)

        tensors = 8 * [None]

        tensors[0] = torch.cat((horizontal_tensor, saliency_map), dim=0)
        tensors[1] = torch.cat((saliency_map, horizontal_tensor), dim=0)
        tensors[2] = torch.cat((vertical_tensor, saliency_map), dim=1)
        tensors[3] = torch.cat((saliency_map, vertical_tensor), dim=1)

        tensors[0] = tensors[0][:width, :]
        tensors[1] = tensors[1][1:, :]
        tensors[2] = tensors[2][:, :height]
        tensors[3] = tensors[3][:, 1:]

        tensors[4] = torch.cat((vertical_tensor, tensors[0]), dim=1)
        tensors[5] = torch.cat((tensors[1], vertical_tensor), dim=1)
        tensors[6] = torch.cat((tensors[2], horizontal_tensor), dim=0)
        tensors[7] = torch.cat((horizontal_tensor, tensors[3]), dim=0)

        tensors[4] = tensors[4][:, :height]
        tensors[5] = tensors[5][:, 1:]
        tensors[6] = tensors[6][1:, :]
        tensors[7] = tensors[7][:width, :]

        for i in range(8):
            saliency_map += tensors[i]

        saliency_map /= 9

        return saliency_map

    # overriding abstract method
    def visualize(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:

        gender_saliency_map, age_saliency_map = self.compute_saliency_map(image, gender, age, use_gpu)
        gender_saliency_map = gender_saliency_map[0]
        age_saliency_map = age_saliency_map[0]

        gender_saliency_map = self._mean_saliency_maps(gender_saliency_map)
        age_saliency_map = self._mean_saliency_maps(age_saliency_map)

        gender_saliency_map = gender_saliency_map.cpu().detach().numpy()
        gender_saliency_fig = px.imshow(gender_saliency_map)
        gender_saliency_fig.update_layout(coloraxis_showscale=False)
        gender_saliency_fig.update_xaxes(showticklabels=False)
        gender_saliency_fig.update_yaxes(showticklabels=False)

        age_saliency_map = age_saliency_map.cpu().detach().numpy()
        age_saliency_fig = px.imshow(age_saliency_map)
        age_saliency_fig.update_layout(coloraxis_showscale=False)
        age_saliency_fig.update_xaxes(showticklabels=False)
        age_saliency_fig.update_yaxes(showticklabels=False)

        return gender_saliency_fig, age_saliency_fig


if __name__ == '__main__':
    device = torch.device('cpu')
    set_seed(42)

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip()])

    loader_train, loader_valid, _ = load_data("./data/UTKFace", num_workers=2,
                                              batch_size=64, transform=transform,
                                              lengths=(0.7, 0.15, 0.15))

    model = CNNClassifier(dim_layers=[32, 64, 128], out_channels=2).to(device)
    model.load_state_dict(torch.load('./models/savedAgeGender/0.01_adam_h_cj9_min_mse_64_[32, 64, '
                                     '128]_0.1_residual=True_maxPool=True.th'))

    object_view = IntegratedSaliencyMap(model)

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
                                                              gender[img_index], age[img_index], use_gpu=False)

            original_fig.show()
            gender_figure.show()
            age_figure.show()

            break
