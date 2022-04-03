import pathlib
from typing import List, Dict, Optional, Tuple

import torch
from torchvision import transforms

from models.utils import load_dict, save_dict


class CNNClassifier(torch.nn.Module):
    """
    Convolutional Neural Net to be used to classify
    """

    class Block(torch.nn.Module):
        """
        Block of the neural net to be repeated multiple times
        """

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                     stride: int = 1, residual: bool = True, conv_layers: int = 3):
            """
            Initialization of a block that composes the classifier
            Architecture is defined here
            :param in_channels: number of channels in the input
            :param out_channels: number of channels produced by the convolution
            :param kernel_size: size of the convolving kernel
            :param stride: stride of the convolution
            :param residual: true if the network should have residual connections
            :param conv_layers: number of convolutional layers (minimum 3)
            """
            super().__init__()
            # Can use max pooling instead of stride
            num_extra_layers = max(conv_layers - 3, 0)
            conv_layers = [
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2),
                                stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2),
                                stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2),
                                stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
            ]
            for k in range(num_extra_layers):
                conv_layers.extend([
                    torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2),
                                    stride=1, bias=False),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(),
                ])
            self.net = torch.nn.Sequential(*conv_layers)

            # Enable residual connections
            self.residual = residual
            if stride != 1 or in_channels != out_channels:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(out_channels),
                )
            else:
                self.downsample = lambda x: x

        def forward(self, x: torch.Tensor):
            """
            Method which runs the given input through the block
            :param x: torch.Tensor((B,C_in,H/stride,W/stride))
            :return: torch.Tensor((B,C_out,H/stride,W/stride))
            """
            if self.residual:
                return self.net(x) + self.downsample(x)
            else:
                return self.net(x)

    def __init__(self, dim_layers: List[int] = [32, 64, 128, 256], in_channels: int = 3, out_channels: int = 1,
                 input_normalization: bool = True, residual: bool = True, in_kernel_size: bool = 7,
                 in_stride: int = 2, max_pooling: bool = True, block_conv_layers: int = 3, **kwargs):
        # flatten_out_layer: bool = False):
        # :param flatten_out_layer: true to use flatten before output
        # linear layer, false to use mean pooling before output layer
        # :param out_mean_pooling: true to use mean pooling for output (only for out_channels of 1)

        """
        Initialization of the classifier
        Architecture is defined here
        :param dim_layers: a list with the number of channels of each convolutional layer
        :param in_channels: number of channels in the input
        :param out_channels: number of channels produced by the convolution
        :param input_normalization: whether to normalize the input or not
        :param residual: true if the network should have residual connections
        :param in_kernel_size: size of the first convolving kernel
        :param in_stride: stride of the first convolution
        :param max_pooling: true to use max pooling between blocks of convolutions,
                            false to use stride in convolutions instead
        :param block_conv_layers: number of convolutional layers of each block (minimum 3)
        """

        super().__init__()
        # stride to use for blocks of convolutions
        stride = 2
        # kernel to use for max pooling if enabled
        max_pool_kernel = 2

        self.normalize = torch.nn.BatchNorm2d(in_channels) if input_normalization else None
        self.out_channels = out_channels

        c = dim_layers[0]
        # Initial convolution
        layers = [
            torch.nn.Conv2d(in_channels, c, kernel_size=in_kernel_size, padding=(in_kernel_size // 2),
                            stride=in_stride, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU(),
        ]
        # Add blocks of convolutions
        for l in dim_layers[1:]:
            layers.append(self.Block(c, l, stride=1 if max_pooling else stride, residual=residual,
                                     conv_layers=block_conv_layers))
            if max_pooling:
                layers.append(torch.nn.MaxPool2d(max_pool_kernel, stride=stride))
            c = l

        self.net = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(c, out_channels)

    def forward(self, x: torch.Tensor):
        """
        Method which runs the given input through the classifier
        :param x: torch.Tensor((B,C_in,H,W))
        :return: torch.Tensor((B,C_out))
        """
        if self.normalize is not None:
            x = self.normalize(x)
        return self.classifier(self.net(x).mean(dim=[2, 3]))


class CNNClassifierTransforms(CNNClassifier):
    def __init__(self, train_transforms=None, pred_transforms=None, *args, **kwargs):
        """
        :param train_transforms:
        :param pred_transforms:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.pred_train = train_transforms if train_transforms is not None else lambda x: x
        self.pred_transforms = pred_transforms if pred_transforms is not None else lambda x: x
        self.to_tensor = transforms.ToTensor()

    def run(self, x):
        if self.training:
            x = self.train_transforms(x)
        else:
            x = self.pred_transforms(x)
        x = self.to_tensor(x)
        return self(x)


MODEL_CLASS = {
    'cnn': CNNClassifier,
    'cnn_t': CNNClassifierTransforms,
}

MODEL_CLASS_KEY = 'model_class'

FOLDER_PATH_KEY = 'path_name'


def save_model(model: torch.nn.Module, folder: str, model_name: str, param_dicts: Dict = None,
               save_model: bool = True) -> None:
    """
    Saves the model so it can be loaded after

    :param model_name: name of the model to be saved (non including extension)
    :param folder: path of the folder where to save the model
    :param param_dicts: dictionary of the model parameters that can later be used to load it
    :param model: model to be saved
    :param save_model: If true the model and dictionary will be saved, otherwise only the dictionary will be saved
    """
    # create folder if it does not exist
    folder_path = f"{folder}/{model_name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    # save model
    if save_model:
        torch.save(model.state_dict(), f"{folder_path}/{model_name}.th")

    # save dict
    if param_dicts is None:
        param_dicts = {}
    # else:  # using pickle instead would solve the problem
    #     param_dicts = param_dicts.copy()
    #     param_dicts[ACTIVATION_KEY] = str(type(param_dicts[ACTIVATION_KEY]).__name__)
    # save model type
    model_class = None
    for k, v in MODEL_CLASS.items():
        if isinstance(model, v):
            model_class = k
            break
    if model_class is None:
        raise Exception("Model class unknown")
    param_dicts[MODEL_CLASS_KEY] = model_class
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict", as_str=True)
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict.pickle", as_str=False)


def load_model(folder_path: pathlib.Path, model_class: Optional[str] = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)

    :param folder_path: folder path of the model to be loaded
    :param model_class: one of the model classes in `MODEL_CLASS` dict. If none, it is obtained from the dictionary
    :return: the loaded model and the dictionary of parameters
    """
    # todo so it does not need to have the same name
    path = f"{folder_path.absolute()}/{folder_path.name}"
    # use pickle instead
    dict_model = load_dict(f"{path}.dict.pickle")

    # get model class
    if model_class is None:
        model_class = dict_model.get(MODEL_CLASS_KEY)

    # set folder path
    dict_model[FOLDER_PATH_KEY] = folder_path.name

    return load_model_data(MODEL_CLASS[model_class](**dict_model), f"{path}.th"), dict_model


def load_model_data(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Loads a model than has been previously saved

    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model
