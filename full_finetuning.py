import math
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn

import json
import random
import torch
from torch.optim.adam import Adam

import plotly.graph_objects as go
from sklearn.cluster import kmeans_plusplus
import pathlib

from utils_for_pipeline import plot_3d_tensor_rel_plotly, load_cuboid_state_dict
from metrics_for_pipeline import calculate_scen_0_rel_sum_cft


# Cuboid Finetuning
 
# Class for trainable model

class CuboidFullFTModel(nn.Module):

    def __init__(
        self,
        nr_of_cuboids: int,
        grid_size: tuple[int, int, int],
        space_range_xy: tuple[int, int],
        space_range_z: tuple[int, int],
        device: str | torch.device,
        seed: int,
        cuboid_state_dict: dict,
    ) -> None:
        """
        Constructor that creates an object of the CuboidFullFTModel class.
        This is the cuboid model that is loaded from a pretraining step and 
        is to be further trained using full fine-tuning based on the path data.
        Parameters of the constructor:
        * nr_of cuboids: How many cuboids are present in the model at the start of fine-tuning (from pretraining)
        * grid_size: How many grid cells per direction, e.g., (10, 10, 10) or (10, 10, 116)
        * space_range_xy: Value ranges of the axes (x, y), e.g., (0, 10)
        * space_range_z: Value ranges of the axes (z), e.g., (0, 10) or (0, 116)
        * device: CPU or GPU (cuda)
        * seed: random seed for reproducibility in random processes
        * cuboid_state_dict: Parameters of the pre-trained cuboids transferred from pre-training (raw values, not squared)
        """
        super().__init__()

        self.nr_of_cuboids = torch.tensor(
            nr_of_cuboids, dtype=torch.int32, requires_grad=False
        )
        self.grid_size = torch.tensor(grid_size, dtype=torch.int32, requires_grad=False)
        self.space_range_xy = torch.tensor(
            space_range_xy, dtype=torch.float32, requires_grad=False
        )
        self.space_range_z = torch.tensor(
            space_range_z, dtype=torch.float32, requires_grad=False
        )
        self.device = torch.device(device)
        print(self.device)

        cell_size_x = (space_range_xy[1] - space_range_xy[0]) / grid_size[0]
        cell_size_y = (space_range_xy[1] - space_range_xy[0]) / grid_size[1]
        cell_size_z = (space_range_z[1] - space_range_z[0]) / grid_size[2]
        self.cell_size = torch.tensor(
            (cell_size_x, cell_size_y, cell_size_z),
            dtype=torch.float32,
            requires_grad=False,
        )
        self.volume_of_a_grid_cell = torch.prod(self.cell_size)

        # Loading the cuboids from the pretraining from the Cuboid State Dictionary
        if cuboid_state_dict is not None:
            self.centers = nn.Parameter(cuboid_state_dict["centers"].to(self.device), requires_grad=True)
            self.radiations = nn.Parameter(cuboid_state_dict["radiations"].to(self.device), requires_grad=True)
            self.half_lengths_x = nn.Parameter(cuboid_state_dict["half_lengths_x"].to(self.device), requires_grad=True)
            self.half_widths_y = nn.Parameter(cuboid_state_dict["half_widths_y"].to(self.device), requires_grad=True)
            self.half_heights_z = nn.Parameter(cuboid_state_dict["half_heights_z"].to(self.device), requires_grad=True)
            self.nr_of_cuboids = torch.tensor(
                cuboid_state_dict["radiations"].shape[0], dtype=torch.int32, device=self.device, requires_grad=False
            )
            
        else: 
            raise ValueError("Fehler bei laden des Cuboid State Dictionaries.")


        self.current_reconstruction = torch.full(grid_size, 0.0, requires_grad=False)

        self.grid_cell_bounds = (
            self.get_grid_bounds() 
        )

        # Variables for saving best loss on training set
        self.best_loss = float("inf") 
        self.best_reconstruction = self.current_reconstruction.clone() 
        self.best_cuboid_config= {
            "centers": self.centers.clone(),
            "radiations": self.radiations.clone(),
            "half_lengths_x": self.half_lengths_x.clone(),
            "half_widths_y": self.half_widths_y.clone(),
            "half_heights_z": self.half_heights_z.clone(),
        }
        
        # Calculation of the initial reconstruction (starting point: Cuboid Prior)
        self.compute_current_reconstruction()
        self.init_reconstruction = self.current_reconstruction.clone().detach()


        # Variables for saving best loss on validation set
        self.validation_best_loss = float("inf")
        self.validation_best_reconstruction = self.current_reconstruction.clone() 
        self.validation_best_cuboid_config= {
            "centers": self.centers.clone(),
            "radiations": self.radiations.clone(),
            "half_lengths_x": self.half_lengths_x.clone(),
            "half_widths_y": self.half_widths_y.clone(),
            "half_heights_z": self.half_heights_z.clone(),
        }

    def get_grid_bounds(self) -> torch.Tensor:
        """
        Function that calculates the min and max coordinates of the grid cells in space (returns a 4D array ((grid_size), 6))
        """
        x_coords = torch.linspace(
            self.space_range_xy[0].item(),
            self.space_range_xy[1].item(),
            steps=int(self.grid_size[0].item()) + 1,
        )
        y_coords = torch.linspace(
            self.space_range_xy[0].item(),
            self.space_range_xy[1].item(),
            steps=int(self.grid_size[1].item()) + 1,
        )
        z_coords = torch.linspace(
            self.space_range_z[0].item(),
            self.space_range_z[1].item(),
            steps=int(self.grid_size[2].item()) + 1,
        )

        grid_cell_bounds = torch.zeros((*self.grid_size, 6), dtype=torch.float32)

        grid_cell_bounds[..., 0] = x_coords[:-1].view(-1, 1, 1).expand(*self.grid_size)
        grid_cell_bounds[..., 1] = x_coords[1:].view(-1, 1, 1).expand(*self.grid_size)
        grid_cell_bounds[..., 2] = y_coords[:-1].view(1, -1, 1).expand(*self.grid_size)
        grid_cell_bounds[..., 3] = y_coords[1:].view(1, -1, 1).expand(*self.grid_size)
        grid_cell_bounds[..., 4] = z_coords[:-1].view(1, 1, -1).expand(*self.grid_size)
        grid_cell_bounds[..., 5] = z_coords[1:].view(1, 1, -1).expand(*self.grid_size)

        return grid_cell_bounds
    
    def freeze_except_shape_xyz(self):
        """ Functions for implementing the optional hyperparameter to first train the shape parameters for a few epochs, then perform full fine-tuning """

        for param in self.parameters():
            param.requires_grad = False

        self.half_lengths_x.requires_grad = True
        self.half_widths_y.requires_grad = True
        self.half_heights_z.requires_grad = True

    def unfreeze_all(self):
        """ Unfreezing of all parameters """
        for param in self.parameters():
            param.requires_grad = True
    

    def plot_cuboids_in_3d_space(self, divisor = 10) -> None:
        """
        Function that plots the current cuboids in 3D space. This is done based on their positions in space and their lengths, widths, and heights.
        The transparency reflects the iodine value. The divisor parameter scales the transparency. Example: divisor = 10 -> transparent if iodine is between 0 and 10, >10 not transparent.
        """
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")

        for i in range(self.centers.size(0)):
            center = self.centers[i].detach().cpu().numpy()
            half_length_x = (self.half_lengths_x[i] ** 2).detach().cpu().item()
            half_width_y = (self.half_widths_y[i] ** 2).detach().cpu().item()
            half_height_z = (self.half_heights_z[i] ** 2).detach().cpu().item()
            opacity = ((self.radiations[i] ** 2)/divisor).detach().cpu().item()

            corners = np.array(
                [
                    [
                        center[0] - half_length_x,
                        center[1] - half_width_y,
                        center[2] - half_height_z,
                    ],
                    [
                        center[0] + half_length_x,
                        center[1] - half_width_y,
                        center[2] - half_height_z,
                    ],
                    [
                        center[0] + half_length_x,
                        center[1] + half_width_y,
                        center[2] - half_height_z,
                    ],
                    [
                        center[0] - half_length_x,
                        center[1] + half_width_y,
                        center[2] - half_height_z,
                    ],
                    [
                        center[0] - half_length_x,
                        center[1] - half_width_y,
                        center[2] + half_height_z,
                    ],
                    [
                        center[0] + half_length_x,
                        center[1] - half_width_y,
                        center[2] + half_height_z,
                    ],
                    [
                        center[0] + half_length_x,
                        center[1] + half_width_y,
                        center[2] + half_height_z,
                    ],
                    [
                        center[0] - half_length_x,
                        center[1] + half_width_y,
                        center[2] + half_height_z,
                    ],
                ]
            )

            faces = [
                [corners[j] for j in [0, 1, 2, 3]],
                [corners[j] for j in [4, 5, 6, 7]],
                [corners[j] for j in [0, 1, 5, 4]],
                [corners[j] for j in [2, 3, 7, 6]],
                [corners[j] for j in [1, 2, 6, 5]],
                [corners[j] for j in [4, 7, 3, 0]],
            ]

            cuboid = Poly3DCollection(
                faces,
                facecolors="cyan",
                alpha=min(1, max(0, opacity)),
                linewidths=1,
                edgecolors="r",
            )

            ax.add_collection3d(cuboid)

        ax.set_box_aspect(
            [
                np.ptp(self.space_range_xy.tolist()),
                np.ptp(self.space_range_xy.tolist()),
                np.ptp(self.space_range_z.tolist()),
            ]
        )

        ax.set_xlim(tuple(self.space_range_xy.tolist()))
        ax.set_ylim(tuple(self.space_range_xy.tolist()))
        ax.set_zlim(tuple(self.space_range_z.tolist()))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()


    def forward(
        self,
        paths_data_batch: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        forward method, takes rasterized data from a batch of paths, returns calculated_exposure of these paths (based on the current reconstruction)
        """
        self.compute_current_reconstruction() 

        batch_calculated_exposures: list[torch.Tensor] = []

        for path_data in paths_data_batch:
            path_protection_values = path_data["protection_values"]
            path_grid_coordinates = path_data["grid_coordinates"]
            path_exposure_values = self.current_reconstruction[
                path_grid_coordinates[:, 0],
                path_grid_coordinates[:, 1],
                path_grid_coordinates[:, 2],
            ]

            calculated_exposure = torch.dot(
                path_protection_values, path_exposure_values
            )

            batch_calculated_exposures.append(calculated_exposure)

        batch_calculated_exposures_stacked = torch.stack(batch_calculated_exposures)

        return batch_calculated_exposures_stacked
    
    def compute_current_reconstruction(self) -> None:
        """
        Function that calculates the current load values in space-time based on the current cuboid configuration (centers, radiations, lengths, widths, heights of the cuboids)
        """
        # Determine the minimum and maximum coordinates (x, y, z) of the cuboids
        cuboid_mins = self.centers - torch.stack(
            [self.half_lengths_x**2, self.half_widths_y**2, self.half_heights_z**2],
            dim=-1,
        )
        cuboid_maxs = self.centers + torch.stack(
            [self.half_lengths_x**2, self.half_widths_y**2, self.half_heights_z**2],
            dim=-1,
        )

        # Load the minimum and maximum coordinates of the grid cells and add another dimension (number of cuboids)
        grid_bounds_expanded = self.grid_cell_bounds.unsqueeze(0).expand(
            int(self.nr_of_cuboids.item()), *self.grid_cell_bounds.shape
        )
        # before: (grid_size_x, grid_size_y, grid_size_z, 6), after: (nr_of_cuboids, grid_size_x, grid_size_y, grid_size_z, 6)

        cuboid_mins = cuboid_mins.to(self.device)
        cuboid_maxs = cuboid_maxs.to(self.device)
        grid_bounds_expanded = grid_bounds_expanded.to(self.device)

        # Calculate the overlaps of the cuboids with the grid cells per dimension (vectorized form, so that all overlaps are calculated simultaneously for time efficiency)
        overlap_x = torch.clamp_min(
            torch.min(cuboid_maxs[:, None, None, None, 0], grid_bounds_expanded[..., 1])
            - torch.max(
                cuboid_mins[:, None, None, None, 0], grid_bounds_expanded[..., 0]
            ),
            0,
        )
        overlap_y = torch.clamp_min(
            torch.min(cuboid_maxs[:, None, None, None, 1], grid_bounds_expanded[..., 3])
            - torch.max(
                cuboid_mins[:, None, None, None, 1], grid_bounds_expanded[..., 2]
            ),
            0,
        )
        overlap_z = torch.clamp_min(
            torch.min(cuboid_maxs[:, None, None, None, 2], grid_bounds_expanded[..., 5])
            - torch.max(
                cuboid_mins[:, None, None, None, 2], grid_bounds_expanded[..., 4]
            ),
            0,
        )

        # Calculate the volume of overlap of each cuboid with each grid cell (vectorized code)
        volume_overlap = overlap_x * overlap_y * overlap_z

        # Normalize the value based on the volume of a single grid cell
        volume_overlap_normalized = volume_overlap / self.volume_of_a_grid_cell

        # Multiply the calculated overlap volumes by the iodine values of the respective cuboids
        radiation_contribution = volume_overlap_normalized * (
            self.radiations[:, None, None, None] ** 2
        )

        # Sum the calculated load values per grid cell and save them as the current reconstruction of the model
        self.current_reconstruction = radiation_contribution.sum(dim=0)


# Class for full finetuning process 

class CuboidFullFTTraining():

    def __init__(
            self,
            cuboid_state_dict: dict,
            device: str | torch.device,
            paths_data_directory: str,
            train_first_index: int,
            train_last_index: int,
            hyperparams: dict,
            adc: bool = False,
            lr_warmup_epochs: int = 0,
    ):
        """
        Constructor that creates an object of the CuboidFullFTTraining class. This involves:
         * specifying the device for training (GPU or CPU),
         * creating an object of the CuboidFullFTModel class, which represents the basic pre-trained Cuboid model (based on the transferred Cuboid State Dict) 
         and moved to the device,
         * prepares the loading of the path data,
         * sets the parameters of the training process (initial learning rate, optimizer, criterion, learning rate scheduler, number of epochs, batch size)
         * adc: Specifies whether or not the adaptive density control method is used in full fine-tuning, default False
         * lr_warmup: Specifies whether or not a learning rate warmup is used in full fine-tuning, default 0 epochs
        """

        if (device == "cuda:0") and (torch.cuda.is_available()):
            self.device = torch.device("cuda:0")
        elif (device == "cuda:1") and (torch.cuda.is_available()):
            self.device = torch.device("cuda:1")    
        else:
            self.device = torch.device("cpu")

        model_cub = CuboidFullFTModel(
            nr_of_cuboids=cuboid_state_dict["radiations"].shape[0],
            grid_size=hyperparams['grid_size'],
            space_range_xy=hyperparams['space_range_xy'],
            space_range_z=hyperparams['space_range_z'],
            device=self.device, 
            seed=hyperparams['seed'],
            cuboid_state_dict=cuboid_state_dict
            )

        self.model = model_cub.to(self.device)

        self.paths_data_directory = paths_data_directory
        self.train_first_index = train_first_index
        self.train_last_index = train_last_index
        self.test_first_index = int(hyperparams["test_first_index"])
        self.test_last_index = int(hyperparams["test_last_index"])
        self.validation_first_index = int(hyperparams["validation_first_index"])
        self.validation_last_index = int(hyperparams["validation_last_index"])

        self.adc = adc #True or False
        self.warmup_epochs = lr_warmup_epochs
        self.only_shape_epochs = 0

        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.lr = hyperparams['initial_lr']
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        # If only the shape parameters are to be trained at the beginning
        if hyperparams['start_w_shape_then_all'] == True:
            self.only_shape_epochs = 19

            self.model.freeze_except_shape_xyz()
            for param in self.model.parameters():
                print(param.requires_grad) 

            self.lr = 0.01
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        # If the learning rates per parameter group from the partial fine-tuning approach are to be applied
        if hyperparams['lr_per_param_from_partial'] == True:

            self.optimizer = Adam([
                {'params': self.model.centers, 'lr': 0.001},
                {'params': self.model.radiations, 'lr': 0.1},
                {'params': self.model.half_lengths_x, 'lr': 0.01},
                {'params': self.model.half_widths_y, 'lr': 0.01},
                {'params': self.model.half_heights_z, 'lr': 0.01}
            ])

        if self.warmup_epochs == 0:

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                "min",
                factor=hyperparams['scheduler']['factor'], 
                patience=hyperparams['scheduler']['patience'], 
                cooldown=hyperparams['scheduler']['cooldown'], 
                threshold=hyperparams['scheduler']['threshold'], 
            )
        elif self.warmup_epochs > 0:

            self.warmup_factor = 0.01 

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self._warmup_lr_lambda)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Learning rate at beginning is {current_lr}")

        else:
            raise Exception("Problem mit dem Warmup.")

        self.batch_size = (
            int(hyperparams['batch_size'])
        )
        self.num_epochs = hyperparams['num_epochs'] 
        self.num_batches = (
            self.train_last_index + 1 - self.train_first_index
        ) // self.batch_size
        torch.manual_seed(hyperparams['torch_seed']) 
        self.hyperparams = hyperparams

    def _warmup_lr_lambda(self, epoch):
        """ Warm-Up Scheduler Funktion. Gibt einen Faktor für die Learning Rate zurück. """
        if epoch < self.warmup_epochs:
            warmup_progress = (epoch + 1) / self.warmup_epochs 
            return self.warmup_factor + (1 - self.warmup_factor) * warmup_progress
        else:
            return 1.0

    def load_pathdataset(
        self,
        split: str = "train",
    ) -> list[dict[str, Any]]:
        """
        Function that loads the path data from its directory at the start of training, stores it in a dictionary,
        and converts the entries into tensors. By default, the train split is loaded, but the validation and test data can also be loaded using the same function.
        split = "train" for the training data, "test" for the test data, "validation" for the validation data
        """

        dir_path = pathlib.Path(self.paths_data_directory)
        data_paths = {}

        if split == "train":
            for i in range(self.train_first_index, self.train_last_index):
                filepath = dir_path / f"{i}.json"

                if filepath.exists():
                    with filepath.open("r") as f:
                        file_data = json.load(f)
                        file_name_without_extension = filepath.stem
                        data_paths[file_name_without_extension] = file_data
                else:
                    print(f"File {filepath} does not exist.")
        elif split == "test":
            for i in range(self.test_first_index, self.test_last_index):
                filepath = dir_path / f"{i}.json"

                if filepath.exists():
                    with filepath.open("r") as f:
                        file_data = json.load(f)
                        file_name_without_extension = filepath.stem
                        data_paths[file_name_without_extension] = file_data
                else:
                    print(f"File {filepath} does not exist.")
        elif split == "validation":
            for i in range(self.validation_first_index, self.validation_last_index):
                filepath = dir_path / f"{i}.json"

                if filepath.exists():
                    with filepath.open("r") as f:
                        file_data = json.load(f)
                        file_name_without_extension = filepath.stem
                        data_paths[file_name_without_extension] = file_data
                else:
                    print(f"File {filepath} does not exist.")

        for v in data_paths.values():
            for key, value in v.items():
                if isinstance(value, list):
                    v[key] = torch.tensor(value).to(self.device)
                elif isinstance(value, int | float):
                    v[key] = torch.tensor([value]).to(self.device)

        paths_data_list = [v for k, v in data_paths.items()]
        return paths_data_list

    def run(
        self,
    ):
        """
        Function that performs the full finetuning process of the cuboid model. This starts the training,
        iterates over the number of epochs to be performed, and returns the best reconstruction achieved at the end.
        In the training process, the parameters of the cuboids are optimized using a gradient method.
        In addition, the adaptive density control method is optionally performed every twenty epochs (deleting and adding in a shift pattern).
        More detailed information can be found in the comments on the individual code sections.
        """

        paths_data_list = self.load_pathdataset()
        validation_paths_data_list = self.load_pathdataset(split="validation")
        print("Training and validation path data loaded.")

        for epoch in range(self.num_epochs):

            #self.model.plot_cuboids_in_3d_space() #Can be inserted at any point in the training code to visualize the current cuboids

            self.model.train()

            random.seed(epoch)
            random.shuffle(paths_data_list)

            for batch_idx in range(self.num_batches):

                batch_start = batch_idx * self.batch_size
                batch_end = (batch_idx + 1) * self.batch_size
                paths_data_batch = paths_data_list[batch_start:batch_end]

                self.optimizer.zero_grad()

                # Load the actual exposure values of the paths used
                true_exposures = [
                    path_data["exposure"] for path_data in paths_data_batch
                ]
                true_exposures_tensor = torch.tensor(
                    true_exposures, dtype=torch.float32, device=self.device
                )

                # Forward Pass: Calculating the exposure values of the paths in the batch based on the current state of the model (the current reconstruction)
                calculated_exposures = self.model(paths_data_batch)

                batch_loss = self.criterion(calculated_exposures, true_exposures_tensor)

                # Backward Pass
                batch_loss.backward()

                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # Epoch loss on the training data
                true_exposures = [
                    path_data["exposure"] for path_data in paths_data_list
                ]
                true_exposures_tensor = torch.tensor(
                    true_exposures, dtype=torch.float32, device=self.device
                )

                calculated_exposures = self.model(paths_data_list)

                epoch_loss = self.criterion(
                    calculated_exposures, true_exposures_tensor
                ).item()

                # Epoch loss on the validation data
                validation_true_exposures = [
                    validation_path_data["exposure"] for validation_path_data in validation_paths_data_list
                ]
                validation_true_exposures_tensor = torch.tensor(
                    validation_true_exposures, dtype=torch.float32, device=self.device
                )

                validation_calculated_exposures = self.model(validation_paths_data_list)

                validation_epoch_loss = self.criterion(
                    validation_calculated_exposures, validation_true_exposures_tensor
                ).item()


            print("Epoch:" + str(epoch))
            print("Training-Loss of this epoch:" + str(epoch_loss))
            print("Validation-Loss of this epoch:" + str(validation_epoch_loss))
            print("Current nr of cuboids:" + str(self.model.nr_of_cuboids))

            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                self.scheduler.step()
            else: 
                self.scheduler.step(epoch_loss)

            current_learning_rates = [param_group['lr'] for param_group in self.optimizer.param_groups]
            print(f"Epoch {epoch}: Learning Rates = {current_learning_rates}")

            if self.warmup_epochs > 0 and epoch == self.warmup_epochs:
                
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                    self.optimizer,
                    "min",
                    factor=self.hyperparams['scheduler']['factor'],
                    patience=self.hyperparams['scheduler']['patience'],
                    cooldown=self.hyperparams['scheduler']['cooldown'],
                    threshold=self.hyperparams['scheduler']['threshold'],
                )

            # Overwrite the best model if the training loss of the current epoch is lower than the best loss achieved previously
            if epoch_loss < self.model.best_loss:
                self.model.best_loss = epoch_loss
                self.model.best_reconstruction = self.model.current_reconstruction.clone().detach()
                self.model.best_cuboid_config = {
                    "centers": self.model.centers.clone().detach(),
                    "radiations": self.model.radiations.clone().detach(),
                    "half_lengths_x": self.model.half_lengths_x.clone().detach(),
                    "half_widths_y": self.model.half_widths_y.clone().detach(),
                    "half_heights_z": self.model.half_heights_z.clone().detach(),
                }

                torch.save(self.model, f"best_fft_model{self.hyperparams['device'][-1]}.pth")

            # Overwrite the best model if the validation loss of the current epoch is lower than the previously achieved best loss
            if validation_epoch_loss < self.model.validation_best_loss:
                self.model.validation_best_loss = validation_epoch_loss
                self.model.validation_best_reconstruction = self.model.current_reconstruction.clone().detach()
                self.model.validation_best_cuboid_config = {
                    "centers": self.model.centers.clone().detach(),
                    "radiations": self.model.radiations.clone().detach(),
                    "half_lengths_x": self.model.half_lengths_x.clone().detach(),
                    "half_widths_y": self.model.half_widths_y.clone().detach(),
                    "half_heights_z": self.model.half_heights_z.clone().detach(),
                }

                torch.save(self.model, f"validation_best_fft_model{self.hyperparams['device'][-1]}.pth")

            if self.hyperparams['start_w_shape_then_all'] == True and epoch == 19:
                # Deactivation of parameter freezing after 20 epochs in which only the shape parameters were trained
                self.model.unfreeze_all()
                for param in self.model.parameters():
                    print(param.requires_grad) 

                current_learning_rates = [0.01]
                self.optimizer = Adam(self.model.parameters(), lr=current_learning_rates[0])  
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        "min",
                        factor=self.hyperparams['scheduler']['factor'],
                        patience=self.hyperparams['scheduler']['patience'],
                        cooldown=self.hyperparams['scheduler']['cooldown'],
                        threshold=self.hyperparams['scheduler']['threshold'],
                    )

            # if the ADC mechanism is not supposed to be performed
            if self.adc == False:
                continue

            # Adaptive Density Control
            # Deletion of cuboids with very low iodine value, outside the area, and with minimal volume
            if epoch % 20 == 15 and epoch > self.warmup_epochs and epoch >= (self.only_shape_epochs+10):
                
                print(f"Anzahl Cuboids vor Löschung irrelevanter Cuboids: {self.model.nr_of_cuboids}")
                
                # Mask for "empty" cuboids
                mask_radiations = self.model.radiations**2 >= 0.001
                print(f"Anzahl leerer Cuboids: {self.model.nr_of_cuboids - mask_radiations.sum().item()}")

                # Mask for cuboids outside the area
                cuboid_mins = self.model.centers - torch.stack(
                    [self.model.half_lengths_x**2, self.model.half_widths_y**2, self.model.half_heights_z**2],
                    dim=-1)

                cuboid_maxs = self.model.centers + torch.stack(
                    [self.model.half_lengths_x**2, self.model.half_widths_y**2, self.model.half_heights_z**2],
                    dim=-1)
                
                relevant_area_min = torch.Tensor([self.model.space_range_xy[0], self.model.space_range_xy[0], self.model.space_range_z[0]])
                relevant_area_max = torch.Tensor([self.model.space_range_xy[1], self.model.space_range_xy[1], self.model.space_range_z[1]])
                
                overlap_x = (cuboid_mins[:, 0] <= relevant_area_max[0]) & (cuboid_maxs[:, 0] >= relevant_area_min[0])
                overlap_y = (cuboid_mins[:, 1] <= relevant_area_max[1]) & (cuboid_maxs[:, 1] >= relevant_area_min[1])
                overlap_z = (cuboid_mins[:, 2] <= relevant_area_max[2]) & (cuboid_maxs[:, 2] >= relevant_area_min[2])

                mask_area = overlap_x & overlap_y & overlap_z
                print(f"Anzahl Cuboids außerhalb Gebiet: {self.model.nr_of_cuboids - mask_area.sum().item()}")

                # Mask for miniature cuboids
                volumes = ((self.model.half_lengths_x**2)*2) * ((self.model.half_widths_y**2)*2) * ((self.model.half_heights_z**2)*2)
                mask_volumes = volumes >= 0.000001
                print(f"Anzahl Miniatur-Cuboids: {self.model.nr_of_cuboids - mask_volumes.sum().item()}")

                # Combination of the three masks
                combined_mask = mask_radiations & mask_area & mask_volumes
                print(f"Anzahl verbleibender Cuboids nach Löschung: {combined_mask.sum().item()}")

                if not combined_mask.all():  # if there is at least 1 False value in the mask
                    
                    self.model.centers = nn.Parameter(
                        self.model.centers[combined_mask].clone().detach().requires_grad_(True)
                    )
                    self.model.radiations = nn.Parameter(
                        self.model.radiations[combined_mask]
                        .clone()
                        .detach()
                        .requires_grad_(True)
                    )
                    self.model.half_lengths_x = nn.Parameter(
                        self.model.half_lengths_x[combined_mask]
                        .clone()
                        .detach()
                        .requires_grad_(True)
                    )
                    self.model.half_widths_y = nn.Parameter(
                        self.model.half_widths_y[combined_mask]
                        .clone()
                        .detach()
                        .requires_grad_(True)
                    )
                    self.model.half_heights_z = nn.Parameter(
                        self.model.half_heights_z[combined_mask]
                        .clone()
                        .detach()
                        .requires_grad_(True)
                    )

                    self.model.nr_of_cuboids = torch.tensor(
                        combined_mask.sum().item(), dtype=torch.int32, requires_grad=False
                    ).to(self.model.centers.device)

                    self.optimizer = Adam(
                        self.model.parameters(), lr=current_learning_rates[0]
                    )

                    if self.hyperparams['lr_per_param_from_partial'] == True:
                        self.optimizer = Adam([
                            {'params': self.model.centers, 'lr': current_learning_rates[0]},
                            {'params': self.model.radiations, 'lr': current_learning_rates[1]},
                            {'params': self.model.half_lengths_x, 'lr': current_learning_rates[2]},
                            {'params': self.model.half_widths_y, 'lr': current_learning_rates[3]},
                            {'params': self.model.half_heights_z, 'lr': current_learning_rates[4]}
                        ])

                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        "min",
                        factor=self.hyperparams['scheduler']['factor'],
                        patience=self.hyperparams['scheduler']['patience'],
                        cooldown=self.hyperparams['scheduler']['cooldown'],
                        threshold=self.hyperparams['scheduler']['threshold'],
                    )

                print(f"Anzahl Cuboids nach Löschung irrelevanter Cuboids: {self.model.nr_of_cuboids}")

            # Adding new cuboids along under-reconstructed paths
            if epoch % 20 == 0 and epoch > self.warmup_epochs and epoch >= (self.only_shape_epochs+10):
                
                new_centers = create_specific_centers_b(
                    paths_data_list=paths_data_list,
                    model=self.model,
                    epoch_seed=epoch,
                )

                new_centers = new_centers.to(self.model.centers.device)
                # initialized and then, if suitable, iodine value trained, otherwise deleted again
                new_radiations = torch.full(
                    (5,), 0.00001, dtype=torch.float32, device=self.model.centers.device
                )
                new_half_lengths_x = torch.full(
                    (5,),
                    math.sqrt(self.hyperparams['initial_length_width_height']),
                    dtype=torch.float32,
                    device=self.model.centers.device,
                )
                new_half_widths_y = torch.full(
                    (5,),
                    math.sqrt(self.hyperparams['initial_length_width_height']),
                    dtype=torch.float32,
                    device=self.model.centers.device,
                )
                new_half_heights_z = torch.full(
                    (5,),
                    math.sqrt(self.hyperparams['initial_length_width_height']),
                    dtype=torch.float32,
                    device=self.model.centers.device,
                )

                self.model.centers = nn.Parameter(
                    torch.cat((self.model.centers, new_centers), dim=0)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                self.model.radiations = nn.Parameter(
                    torch.cat((self.model.radiations, new_radiations), dim=0)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                self.model.half_lengths_x = nn.Parameter(
                    torch.cat((self.model.half_lengths_x, new_half_lengths_x), dim=0)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                self.model.half_widths_y = nn.Parameter(
                    torch.cat((self.model.half_widths_y, new_half_widths_y), dim=0)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                self.model.half_heights_z = nn.Parameter(
                    torch.cat((self.model.half_heights_z, new_half_heights_z), dim=0)
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )

                self.model.nr_of_cuboids = torch.tensor(
                    self.model.nr_of_cuboids.item() + 5,
                    dtype=torch.int32,
                    requires_grad=False,
                ).to(self.model.centers.device)

                self.optimizer = Adam(
                    self.model.parameters(), lr=current_learning_rates[0]
                )
                
                if self.hyperparams['lr_per_param_from_partial'] == True:

                    self.optimizer = Adam([
                        {'params': self.model.centers, 'lr': current_learning_rates[0]},
                        {'params': self.model.radiations, 'lr': current_learning_rates[1]},
                        {'params': self.model.half_lengths_x, 'lr': current_learning_rates[2]},
                        {'params': self.model.half_widths_y, 'lr': current_learning_rates[3]},
                        {'params': self.model.half_heights_z, 'lr': current_learning_rates[4]}
                    ])
                
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    "min",
                    factor=self.hyperparams['scheduler']['factor'],
                    patience=self.hyperparams['scheduler']['patience'],
                    cooldown=self.hyperparams['scheduler']['cooldown'],
                    threshold=self.hyperparams['scheduler']['threshold'],
                )

            # Avoid accumulation of unused data in the GPU cache
            if epoch % 10 == 9:
                torch.cuda.empty_cache()

        best_model = torch.load(
            f"best_fft_model{self.hyperparams['device'][-1]}.pth"
        ) 
        print("Bestes Modell wieder geladen.")

        validation_best_model = torch.load(
            f"validation_best_fft_model{self.hyperparams['device'][-1]}.pth"
        ) 
        print("Bestes Validation-Modell wieder geladen.")

        return (self.model.best_cuboid_config, self.model.best_reconstruction, best_model, self.model.validation_best_cuboid_config, self.model.validation_best_reconstruction, validation_best_model) #T-Cuboid State Dictionary, T-Reconstruction, T-Model (CuboidFullFTModel), V-Cuboid State Dictionary, V-Reconstruction, V-Model (CuboidFullFTModel)
        
    def visualize_reconstruction(
            self,
    ) -> None:
        """
        Visualization function for the reconstruction.
        """
        resulting_reconstruction = self.model.best_reconstruction
        plot_3d_tensor_rel_plotly(
            resulting_reconstruction,
            filter=self.hyperparams['filter'], 
            timesteps=self.hyperparams['timesteps'], 
            space_range_xy=self.model.space_range_xy[1],
        ).write_html("best_reconstruction_vis_fft.html")


# Auxiliary functions for the ADC mechanism

def get_true_exposures(paths_data_list: list[dict[str, Any]]) -> torch.Tensor:
    """
    Helper function to output the actual exposure values of the paths.
    """
    true_exposures = [path_data["exposure"] for path_data in paths_data_list]
    true_exposures_tensor = torch.tensor(true_exposures, dtype=torch.float32)
    return true_exposures_tensor


def get_calculated_exposures(
    paths_data_list: list[dict[str, Any]],
    model: CuboidFullFTModel,
) -> torch.Tensor:
    """
    Helper function to output the exposure values of the paths calculated according to the current state of the model (the current reconstruction).
    """
    with torch.no_grad():
        calculated_exposures = model(paths_data_list)
    return torch.as_tensor(calculated_exposures)


def create_specific_centers_b(
    paths_data_list: list[dict[str, Any]],
    model: CuboidFullFTModel,
    epoch_seed: int,
) -> torch.Tensor:
    """
    Identifies the 10 most under-reconstructed paths. Selects 5 random cells from the grid cells contained in these paths and defines new cuboids centered on these cells.
    """
    random.seed(epoch_seed)
    true_exposures_cpu = get_true_exposures(paths_data_list=paths_data_list)

    calc_exposures = get_calculated_exposures(
        paths_data_list=paths_data_list, model=model
    )
    calc_exposures_cpu = calc_exposures.cpu()

    differences = true_exposures_cpu - calc_exposures_cpu

    top20_values, top20_indices = torch.topk(differences, 10)

    coordinates_list = [
        paths_data_list[index.item()]["grid_coordinates"] for index in top20_indices
    ]
    concatenated_tensor = torch.cat(coordinates_list, dim=0)

    tuple_list = [tuple(row.tolist()) for row in concatenated_tensor]

    random_selection = random.sample(tuple_list, 5)

    add_to_position = model.cell_size * 0.5
    result_tuples = [
        tuple((torch.tensor(t) + add_to_position).tolist()) for t in random_selection
    ]

    new_centers_specific = torch.tensor(result_tuples).to(model.centers.device)

    return new_centers_specific
