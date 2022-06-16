from typing import List

import torch

from models.utils import load_dict


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


# class EnsemblerClassifier(torch.nn.Module):
#     def __init__(self):
#         pass
#
#     def forward(self, x:torch.Tensor):
#         pass


model_factory = {
    'cnn': CNNClassifier,
    # 'ensembler': EnsemblerClassifier,
}


def save_model(model: torch.nn.Module, filename: str = None) -> None:
    """
    Saves the model so it can be loaded after
    :param filename: filename where the model should be saved (non including extension)
    :param model: model to be saved
    """
    from torch import save
    from os import path

    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), f"{filename if filename is not None else n}.th")
    raise Exception(f"Model type {type(model)} not supported")


def load_model_from_name(model_name):
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)
    :param model_name: name of the model to load
    :return: the loaded model
    """
    dict_model = load_dict(f"{model_name}.dict")
    return load_model(f"{model_name}.th", CNNClassifier(**dict_model))


def load_model(model_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Loads a model than has been previously saved
    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    from torch import load
    model.load_state_dict(load(model_path, map_location='cpu'))
    return model
