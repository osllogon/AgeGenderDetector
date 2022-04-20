from abc import ABC, abstractmethod
import plotly
import plotly.express as px
import torch


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

    @torch.no_grad()
    def compute_heat_maps(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool =True) \
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

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        # Creation of probabilities and average tensor with the same dimensions as the image
        losses_gender = torch.zeros_like(image).to(device)
        average_tensor_gender = torch.zeros_like(image).to(device)
        losses_age = torch.zeros_like(image).to(device)
        average_tensor_age = torch.zeros_like(image).to(device)

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
                losses_gender[0, :, i:i + self.mask_size, j:j + self.mask_size] += output[0, 0]
                average_tensor_gender[0, :, i:i + self.mask_size, j:j + self.mask_size] += 1

                # Actualize probabilities and average tensor
                losses_age[0, :, i:i + self.mask_size, j:j + self.mask_size] += output[0, 1]
                average_tensor_age[0, :, i:i + self.mask_size, j:j + self.mask_size] += 1

        # Return 1 - probabilities/average due to you want the zone with a higher drop in probability as a hot zone
        return 1 - (losses_gender / average_tensor_gender), 1 - (losses_age / average_tensor_age)

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

        heat_maps = self.compute_heat_maps(image, gender, age, use_gpu)

        gender_fig = px.imshow(heat_maps[0][0, 0].cpu().detach().numpy())
        gender_fig.update_layout(coloraxis_showscale=False)
        gender_fig.update_xaxes(showticklabels=False)
        gender_fig.update_yaxes(showticklabels=False)

        age_fig = px.imshow(heat_maps[1][0, 0].cpu().detach().numpy())
        age_fig.update_layout(coloraxis_showscale=False)
        age_fig.update_xaxes(showticklabels=False)
        age_fig.update_yaxes(showticklabels=False)

        return gender_fig, age_fig


class GradCAM(Visualization):

    def __init__(self, model: torch.nn.Module, threshold: float = 0) -> None:
        """
        Constructor SaliencyMap class

        Parameters
        ----------
        model : torch.nn.Module
            model for inference
        threshold : float, Optional
            threshold for filtering low values. Default: 0

        Returns
        -------
        None
        """

        self.model = model
        self.threshold = threshold

    @torch.no_grad()
    def compute_grad_cams(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool) -> \
            tuple[torch.Tensor, torch.Tensor]:

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        # pass everything to device
        image = image.to(device)
        self.model = self.model.to(device)

        # prepare device
        inputs = image.clone()
        activations = self.model.net(inputs)
        activations.requires_grad_(True)
        self.model.zero_grad()

        with torch.enable_grad():
            # clear previous gradients and compute new ones
            output = self.model.classifier(activations.mean(dim=[2, 3]))
            output[0, 0].backward()

        weights = activations.grad.sum(dim=[2, 3], keepdims=True)
        gender_cam = (activations*weights).sum(dim=1)

        # create and normalize gender saliency map
        gender_cam = (gender_cam - gender_cam.min()) / (gender_cam.max() - gender_cam.min())
        gender_cam = torch.nn.functional.interpolate(gender_cam.unsqueeze(1), size=(image.size(2), image.size(3)),
                                                     mode='bilinear')

        # zero gradients
        self.model.zero_grad()
        activations.grad.zero_()

        with torch.enable_grad():
            # clear previous gradients and compute new ones
            output = self.model.classifier(activations.mean(dim=[2, 3]))
            output[0, 1].backward()

        weights = activations.grad.sum(dim=[2, 3], keepdims=True)
        age_cam = (activations * weights).sum(dim=1)

        # create and normalize gender saliency map
        age_cam = (age_cam - age_cam.min()) / (age_cam.max() - age_cam.min())
        age_cam = torch.nn.functional.interpolate(age_cam.unsqueeze(1), size=(image.size(2), image.size(3)),
                                                  mode='bilinear')

        return gender_cam[0, 0], age_cam[0, 0]

    # overriding abstract method
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

        heat_maps = self.compute_grad_cams(image, gender, age, use_gpu)

        gender_fig = px.imshow(heat_maps[0].cpu().detach().numpy())
        gender_fig.update_layout(coloraxis_showscale=False)
        gender_fig.update_xaxes(showticklabels=False)
        gender_fig.update_yaxes(showticklabels=False)

        age_fig = px.imshow(heat_maps[1].cpu().detach().numpy())
        age_fig.update_layout(coloraxis_showscale=False)
        age_fig.update_xaxes(showticklabels=False)
        age_fig.update_yaxes(showticklabels=False)

        return gender_fig, age_fig



class SaliencyMap:
    """
    Saliency map class

    Attributes
    ----------
    model : torch.nn.Module
        model for inference
    threshold : float
        threshold for filtering low values

    Methods
    -------
    compute_saliency_map -> torch.Tensor
    visualize -> tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]
    """

    def __init__(self, model: torch.nn.Module, threshold: float = 0) -> None:
        """
        Constructor SaliencyMap class

        Parameters
        ----------
        model : torch.nn.Module
            model for inference
        threshold : float, Optional
            threshold for filtering low values. Default: 0

        Returns
        -------
        None
        """

        self.model = model
        self.threshold = threshold

    @torch.no_grad()
    def compute_saliency_map(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool) -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        This method creates saliency maps

        Parameters
        ----------
        image : torch.Tensor
            image. Dimensions: [1, channels, height, width]
        gender : int
            gender label
        age : float
            age label
        use_gpu : bool
            if true and is available a gpu is used

        Returns
        -------
        torch.Tensor
            gender saliency map. Dimensions: [height, width]
        torch.Tensor
            age saliency map. Dimensions: [height, width]
        """

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        # pass everything to device
        image = image.to(device)
        self.model = self.model.to(device)

        # prepare device
        input_ = image.clone()
        input_.requires_grad_()
        self.model.zero_grad()

        with torch.enable_grad():
            # pass the image through the model
            output = self.model(input_)

            # clear previous gradients and compute new ones
            self.model.zero_grad()
            output[0, 0].backward()

            # create and normalize gender saliency map
            gender_saliency_map, _ = torch.max(torch.abs(input_.grad), dim=1)
            gender_saliency_map = (gender_saliency_map - gender_saliency_map.min()) / \
                                  (gender_saliency_map.max() - gender_saliency_map.min())
            gender_saliency_map[gender_saliency_map < self.threshold] = 0

            # pass the input through the model again
            output = self.model(input_)

            # clear previous gradients and compute new ones
            self.model.zero_grad()
            input_.grad.zero_()
            output[0, 1].backward()

            # create and normalize age saliency map
            age_saliency_map, _ = torch.max(torch.abs(input_.grad), dim=1)
            age_saliency_map = (age_saliency_map - age_saliency_map.min()) / (
                    age_saliency_map.max() - age_saliency_map.min())
            age_saliency_map[age_saliency_map < self.threshold] = 0

            return gender_saliency_map, age_saliency_map

    def visualize(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
            tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
        """
        This methods generates a saliency map, creates a visualization of it and saves it

        Attributes
        ----------
        image : torch.Tensor
            This is the image we use as input. Dimensions: [1, channels, rows, columns]
        gender : str
            gender label
        age : float
            age gender

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


class SaliencyMapCombined(SaliencyMap):

    def __init__(self, model: torch.nn.Module, threshold: float = 0):
        super().__init__(model, threshold)

    # override method
    @torch.no_grad()
    def compute_saliency_map(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool) -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        This method computes combined saliency maps
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
        Returns
        -------
            batch of combined saliency maps. Dimensions: [batch, height, width]
        """

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        # pass everything to device
        image = image.to(device)
        self.model = self.model.to(device)

        # prepare input
        input_ = image.clone()
        input_.requires_grad_()

        with torch.enable_grad():
            # clear previous gradients and compute new ones
            output = self.model(input_)
            self.model.zero_grad()
            output[0, 0].backward()

        # create and normalize gender saliency map
        gender_saliency_map, _ = torch.max(torch.abs(input_.grad)*input_, dim=1)
        gender_saliency_map = (gender_saliency_map - gender_saliency_map.min()) / \
                              (gender_saliency_map.max() - gender_saliency_map.min())
        gender_saliency_map[gender_saliency_map < self.threshold] = 0

        # zero gradients
        self.model.zero_grad()
        input_.grad.zero_()

        with torch.enable_grad():
            # pass the input through the model again
            output = self.model(input_)

            # clear previous gradients and compute new ones
            self.model.zero_grad()
            input_.grad.zero_()
            output[0, 1].backward()

        # create and normalize age saliency map
        age_saliency_map, _ = torch.max(torch.abs(input_.grad)*input_, dim=1)
        age_saliency_map = (age_saliency_map - age_saliency_map.min()) / (
                age_saliency_map.max() - age_saliency_map.min())
        age_saliency_map[age_saliency_map < self.threshold] = 0

        return gender_saliency_map, age_saliency_map


class AverageSaliencyMap(SaliencyMap):

    def __init__(self, model: torch.nn.Module, threshold: float = 0):
        super().__init__(model, threshold)

    def _mean_saliency_maps(self, saliency_map: torch.Tensor, device: torch.device) -> torch.Tensor:
        height = saliency_map.size(2)
        width = saliency_map.size(1)

        horizontal_tensor = torch.zeros(1, 1, height).to(device)
        vertical_tensor = torch.zeros(1, width, 1).to(device)

        tensors = 8 * [None]

        tensors[0] = torch.cat((horizontal_tensor, saliency_map), dim=1)
        tensors[1] = torch.cat((saliency_map, horizontal_tensor), dim=1)
        tensors[2] = torch.cat((vertical_tensor, saliency_map), dim=2)
        tensors[3] = torch.cat((saliency_map, vertical_tensor), dim=2)

        tensors[0] = tensors[0][:, :width, :]
        tensors[1] = tensors[1][:, 1:, :]
        tensors[2] = tensors[2][:, :, :height]
        tensors[3] = tensors[3][:, :, 1:]

        tensors[4] = torch.cat((vertical_tensor, tensors[0]), dim=2)
        tensors[5] = torch.cat((tensors[1], vertical_tensor), dim=2)
        tensors[6] = torch.cat((tensors[2], horizontal_tensor), dim=1)
        tensors[7] = torch.cat((horizontal_tensor, tensors[3]), dim=1)

        tensors[4] = tensors[4][:, :, :height]
        tensors[5] = tensors[5][:, :, 1:]
        tensors[6] = tensors[6][:, 1:, :]
        tensors[7] = tensors[7][:, :width, :]

        for i in range(8):
            saliency_map += tensors[i]

        saliency_map /= 9

        return saliency_map

    # overriding abstract method
    def compute_saliency_map(self, image: torch.Tensor, gender: int, age: float, use_gpu: bool = True) -> \
            tuple[torch.Tensor, torch.Tensor]:

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        gender_saliency_map, age_saliency_map = super().compute_saliency_map(image, gender, age, use_gpu)
        gender_saliency_map = self._mean_saliency_maps(gender_saliency_map, device)
        age_saliency_map = self._mean_saliency_maps(age_saliency_map, device)

        return gender_saliency_map, age_saliency_map