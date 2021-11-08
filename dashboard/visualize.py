from typing import List, Tuple

import numpy as np
import pandas
import plotly.graph_objects as go
import torch
from pandas import DataFrame
from plotly.subplots import make_subplots

from models.models import CNNClassifier, load_model
from models.utils import load_data, LABEL_GENDER, load_dict, accuracy

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


def visualize_data_images(datasets):
    # Show multiple random photos
    number_images = 4
    fig = go.Figure()
    for k in range(number_images):
        # pytorch tensor to numpy

        fig.add_trace(
            go.Image(
                # z=
            )
        )


def visualize_training():
    # Show training data with filters (as tensorboard)
    pass


def visualize_model(model_name, data_path, num_workers: int = 0, batch_size: int = 64,
                    use_gpu: bool = True) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Visualize accuracy and RMSE per dataset and per age
    :param model_name: name of the model to analyze results
    :param data_path: path to the data images
    :param num_workers: number of workers (processes) to use for data loading
    :param batch_size: size of batches to use
    :param use_gpu: whether to use the gpu (true) or cpu (false)
    :return: tuple of plotly figures
    """
    loaders = load_data(data_path, num_workers=num_workers, batch_size=batch_size, transform=None)
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # Load model and loss
    model = load_model(f"{model_name}.th", CNNClassifier(**(load_dict(f"{model_name}.dict")))).to(device)
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

    del model
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


def visualize_fun(model_name, img_path):
    # Show option of adding image and running the model
    # Mostrar opcion de añadir imagen y que corra el modelo sobre ella
    pass


if __name__ == '__main__':
    visualize_model('models/savedAgeGender/0.01_adam_h_min_loss_64_[8, 16, 32, 64, 128]_0.1_residual=True_maxPool=True',
                    "./data/UTKFace")[2].show()
