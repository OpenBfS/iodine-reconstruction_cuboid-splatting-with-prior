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

# Cuboid Pretraining

# Class for trainable model

class CuboidPriorModel(nn.Module):

    def __init__(
        self,
        nr_of_cuboids: int,
        grid_size: tuple[int, int, int],
        space_range_xy: tuple[int, int],
        space_range_z: tuple[int, int],
        initial_radiation: float,
        initial_length_width_height: float,
        device: str | torch.device,
        seed: int,
        cuboid_state_dict: dict | None,
    ) -> None:
        """
        Constructor that creates an object of the CuboidPriorModel class.
        This is the trainable model based on a RODOS calculation.  It translates a prior scenario into an initial cuboid distribution.
        Parameters of the constructor:
        * nr_of cuboids: How many cuboids are to be generated, e.g., 20
        * grid_size: How many grid cells per direction, e.g., (10, 10, 10) or (10, 10, 115)
        * space_range_xy: Value ranges of the axes (x, y), e.g., (0, 10)
        * space_range_z: Value ranges of the axes (z), e.g., (0, 10) or (0, 115)
        * initial_radiation: Initial load value of the cuboids, e.g., 0.1
        * initial_length_width_height: Initial (half) length, width, and height of the cuboids
        * device: CPU or GPU (cuda)
        * seed: Random seed for reproducibility in random processes
        * cuboid_state_dict: Transferred cuboid initialization (optional)
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

        cell_size_x = (space_range_xy[1] - space_range_xy[0]) / grid_size[0]
        cell_size_y = (space_range_xy[1] - space_range_xy[0]) / grid_size[1]
        cell_size_z = (space_range_z[1] - space_range_z[0]) / grid_size[2]
        self.cell_size = torch.tensor(
            (cell_size_x, cell_size_y, cell_size_z),
            dtype=torch.float32,
            requires_grad=False,
        )
        self.volume_of_a_grid_cell = torch.prod(self.cell_size)

        if cuboid_state_dict is not None: 

            self.centers = nn.Parameter(cuboid_state_dict["centers"], requires_grad=True)
            self.radiations = nn.Parameter(cuboid_state_dict["radiations"], requires_grad=True)
            self.half_lengths_x = nn.Parameter(cuboid_state_dict["half_lengths_x"], requires_grad=True)
            self.half_widths_y = nn.Parameter(cuboid_state_dict["half_widths_y"], requires_grad=True)
            self.half_heights_z = nn.Parameter(cuboid_state_dict["half_heights_z"], requires_grad=True)
            self.nr_of_cuboids = torch.tensor(
                cuboid_state_dict["radiations"].shape[0], dtype=torch.int32, requires_grad=False
            )
            
        else: 

            initial_centers = self.initialize_centers_in_3d_space_kmeanspp(
                nr_of_cuboids=nr_of_cuboids, seed=seed
            )

            self.centers = nn.Parameter(initial_centers, requires_grad=True)
            self.radiations = nn.Parameter(
                torch.full((nr_of_cuboids,), initial_radiation, dtype=torch.float32),
                requires_grad=True,
            )
            self.half_lengths_x = nn.Parameter(
                torch.full(
                    (nr_of_cuboids,),
                    math.sqrt(initial_length_width_height),
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )
            self.half_widths_y = nn.Parameter(
                torch.full(
                    (nr_of_cuboids,),
                    math.sqrt(initial_length_width_height),
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )
            self.half_heights_z = nn.Parameter(
                torch.full(
                    (nr_of_cuboids,),
                    math.sqrt(initial_length_width_height),
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )

        self.current_prior_reconstruction = torch.full(grid_size, 0.0, requires_grad=False)

        self.grid_cell_bounds = (
            self.get_grid_bounds()
        )

        self.best_loss = float("inf")
        self.best_prior_reconstruction = self.current_prior_reconstruction.clone()
        self.best_prior_config= {
            "centers": self.centers.clone(),
            "radiations": self.radiations.clone(),
            "half_lengths_x": self.half_lengths_x.clone(),
            "half_widths_y": self.half_widths_y.clone(),
            "half_heights_z": self.half_heights_z.clone(),
        }
        self.init_prior_reconstruction = self.current_prior_reconstruction.clone()
    
    def initialize_centers_in_3d_space_random(
        self,
        nr_of_cuboids: int,
        seed: int,
    ) -> torch.Tensor:
        """
        Function that randomly initializes the centers of the cuboids in 3D space. Old version without k-means++.
        """
        if seed is not None:
            torch.manual_seed(seed)

        min_val_xy, max_val_xy = self.space_range_xy
        min_val_z, max_val_z = self.space_range_z
        random_positions_xy = (
            torch.rand((nr_of_cuboids, 2)) * (max_val_xy - min_val_xy) + min_val_xy
        )
        random_positions_z = (
            torch.rand(nr_of_cuboids) * (max_val_z - min_val_z) + min_val_z
        )
        random_positions = torch.cat(
            (random_positions_xy, random_positions_z.unsqueeze(1)), dim=1
        )

        return random_positions
    
    def initialize_centers_in_3d_space_kmeanspp(
        self,
        nr_of_cuboids: int,
        seed: int,
    ) -> torch.Tensor:
        """
        Function that randomly initializes the centers of the cuboids in 3D space using the kmeans++ algorithm, 
        so that they are well distributed throughout the space.
        """
        if seed is not None:
            np.random.seed(seed)

        min_val_xy, max_val_xy = self.space_range_xy.cpu().numpy()
        min_val_z, max_val_z = self.space_range_z.cpu().numpy()

        random_points_xy = (
        np.random.rand(10000, 2) * (max_val_xy - min_val_xy) + min_val_xy
        )
        random_points_z = (
            np.random.rand(10000) * (max_val_z - min_val_z) + min_val_z
        )
        random_points = np.concatenate(
            (random_points_xy, random_points_z[:, np.newaxis]), axis=1
        )

        centers, _ = kmeans_plusplus(random_points, n_clusters=nr_of_cuboids, random_state=seed)

        centers_tensor = torch.tensor(centers, dtype=torch.float32)
        return centers_tensor
    

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
    

    def get_cuboid_min_max(
        self,
        cuboid_index: int,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Function that can determine the min and max coordinates of a cuboid (based on center and length, width, height)
        """
        center = self.centers[cuboid_index]
        half_lengths = torch.stack(
            [
                self.half_lengths_x[cuboid_index] ** 2,
                self.half_widths_y[cuboid_index] ** 2,
                self.half_heights_z[cuboid_index] ** 2,
            ]
        )

        cuboid_min = center - half_lengths
        cuboid_max = center + half_lengths

        return (
            *cuboid_min,
            *cuboid_max,
        )  # form: cuboid_min_x, cuboid_min_y, cuboid_min_z, cuboid_max_x, cuboid_max_y, cuboid_max_z
    

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
    ) -> torch.Tensor:
        """
        The forward method calculates the current prior translation based on the current cuboid parameters and returns it.
        This function, which calculates the current iodine values in space-time based on the current cuboid configuration (centers, radiations, lengths, widths, heights of the cuboids),
        has been adapted so that the prior scenario, rather than the path data, is used as the training data for this pretraining process. 
        Accordingly, the output is not reconstructed thyroid measurements but the “reconstructed” prior iodine values.
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
        self.current_prior_reconstruction = radiation_contribution.sum(dim=0)
        
        return self.current_prior_reconstruction
    

# Class for pretraining process 

class CuboidPriorTrainingModel():

    def __init__(
            self,
            prior_data_directory: str,
            cuboid_state_dict: dict | None,
            device: str | torch.device,
            hyperparams: dict,
    ):
        """
        Constructor that creates an object of the CuboidPriorTrainingModel class. This involves:
        * specifying the device for training (GPU or CPU),
        * creating an object of the CuboidPriorModel class, which represents the basic Cuboid Prior model to be trained and moved to the device,
        * prepares the loading of the prior scenario,
        * sets the parameters of the training process (initial learning rate, optimizer, criterion, learning rate scheduler, number of epochs, batch size)
        """

        if (device == "cuda:0") and (torch.cuda.is_available()):
            self.device = torch.device("cuda:0")
        elif (device == "cuda:1") and (torch.cuda.is_available()):
            self.device = torch.device("cuda:1")      
        else:
            self.device = torch.device("cpu")

        model_cub_prior = CuboidPriorModel(
            nr_of_cuboids=hyperparams['nr_of_cuboids'],
            grid_size=hyperparams['grid_size'],
            space_range_xy=hyperparams['space_range_xy'],
            space_range_z=hyperparams['space_range_z'],
            initial_radiation=hyperparams['initial_radiation'],
            initial_length_width_height=hyperparams['initial_length_width_height'],
            device=self.device,
            seed=hyperparams['seed'],
            cuboid_state_dict=cuboid_state_dict,
        )

        self.model = model_cub_prior.to(self.device)

        self.prior_data_directory = prior_data_directory

        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.lr = hyperparams['initial_lr']
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=hyperparams['scheduler']['factor'],
            patience=hyperparams['scheduler']['patience'],
            cooldown=hyperparams['scheduler']['cooldown'],
            threshold=hyperparams['scheduler']['threshold'],
        )

        self.num_epochs = int(hyperparams['num_epochs'])
        torch.manual_seed(hyperparams['torch_seed'])
        self.hyperparams = hyperparams

    def load_prior_scenario(
            self,
    ) -> torch.Tensor:
        """
        Function that loads the prior scenario from its directory at the beginning of training, stores it in a tensor, and moves it to the training device.
        """

        dir_path = self.prior_data_directory
        prior_scenario = np.load(dir_path)
        prior_scenario_tensor = torch.from_numpy(prior_scenario).to(dtype=torch.float32)
        prior_scenario_tensor.to(self.device)

        return prior_scenario_tensor

    def run(
        self,
    ) -> tuple[dict, torch.Tensor, CuboidPriorModel]:
        """
        Function that performs the training process of the cuboid prior model. This starts the pretraining,
        iterates over the number of epochs to be performed, and returns the best cuboid prior configuration achieved (as a state dictionary) at the end.
        In the training process, the parameters of the cuboids are optimized using gradient descent.
        In addition, the adaptive density control method is performed every x epochs (frequency configurable) (deleting and adding shifted by 5 epochs).
        More detailed information can be found in the comments on the individual code sections.
        """      

        original_prior = self.load_prior_scenario()
        original_prior = original_prior.to(self.device)

        #self.model.plot_cuboids_in_3d_space() #Can be inserted at any point in the training code to visualize the current cuboids
        epoch = None
        try:
            for epoch in range(self.num_epochs):

                #self.model.plot_cuboids_in_3d_space()

                self.model.train()
                self.optimizer.zero_grad()

                # Forward Pass
                current_prior = self.model()
                current_prior = current_prior.to(self.device)

                epoch_loss = self.criterion(current_prior, original_prior) 

                # Backward Pass
                epoch_loss.backward()

                self.optimizer.step()

                self.model.eval()

                print("Epoch:" + str(epoch))
                print("Loss of this epoch:" + str(epoch_loss))
                print("Current nr of cuboids:" + str(self.model.nr_of_cuboids))

                self.scheduler.step(epoch_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Learning rate at epoch {epoch} is {current_lr}")


                # Overwrite the best model if the loss of the current epoch is lower than the best loss achieved previously
                if epoch_loss < self.model.best_loss:
                    self.model.best_loss = epoch_loss
                    self.model.best_prior_reconstruction = (self.model.current_prior_reconstruction.clone().detach())
                    self.model.best_prior_config = {
                        "centers": self.model.centers.clone().detach(),
                        "radiations": self.model.radiations.clone().detach(),
                        "half_lengths_x": self.model.half_lengths_x.clone().detach(),
                        "half_widths_y": self.model.half_widths_y.clone().detach(),
                        "half_heights_z": self.model.half_heights_z.clone().detach(),
                    }

                    torch.save(self.model, f"best_prior_cuda{self.hyperparams['device'][-1]}.pth")

                if epoch == 0: # saving initial prior reconstruction after initialization
                    self.model.init_prior_reconstruction = (self.model.current_prior_reconstruction.clone().detach())

                # ADC mechanism (frequency configurable, recommended: every 20 epochs)
                # Deletion of cuboids with very low iodine value, outside the area, and with minimal volume
                if epoch % int(self.hyperparams["adc_frequency"]) == (int(self.hyperparams["adc_frequency"]) - 5) and epoch > 0:
                    
                    print(f"Anzahl Cuboids vor Löschung irrelevanter Cuboids: {self.model.nr_of_cuboids}")

                    # Mask for "empty" cuboids
                    mask_radiations = (self.model.radiations**2) >= 0.001
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
                            self.model.parameters(), lr=current_lr
                        )
                        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer,
                            "min",
                            factor=self.hyperparams['scheduler']['factor'],
                            patience=self.hyperparams['scheduler']['patience'],
                            cooldown=self.hyperparams['scheduler']['cooldown'],
                            threshold=self.hyperparams['scheduler']['threshold'],
                        )

                    print(f"Anzahl Cuboids nach Löschung irrelevanter Cuboids: {self.model.nr_of_cuboids}")

                # Add new cuboids to under-reconstructed areas of the prior scenario 
                if epoch % int(self.hyperparams["adc_frequency"]) == 0 and epoch > 0: 

                    add_nr_cuboids = int(self.hyperparams["adc_add_nr"])

                    try: 
                        new_centers, new_radiations, new_half_lengths_x, new_half_widths_y, new_half_heights_z = new_cuboids_underreconstructed_cells(original_prior=original_prior, current_prior=self.model.current_prior_reconstruction, epoch_seed=epoch, device=self.model.device, epoch=epoch, nr_of_cuboids=add_nr_cuboids, new_half_size = self.hyperparams["new_half_size"])
                    except ValueError:
                        print("Negative Differenz unter den Top 20. Training wird beendet.")
                        break

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
                        self.model.nr_of_cuboids.item()+add_nr_cuboids, dtype=torch.int32, requires_grad=False
                    ).to(self.model.centers.device)

                    self.optimizer = Adam(
                        self.model.parameters(), lr=current_lr
                    )
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        "min",
                        factor=self.hyperparams['scheduler']['factor'], 
                        patience=self.hyperparams['scheduler']['patience'], 
                        cooldown=self.hyperparams['scheduler']['cooldown'], 
                        threshold=self.hyperparams['scheduler']['threshold'], 
                    )

                    print(f"Anzahl Cuboids nach Einstreuung neuer Cuboids: {self.model.nr_of_cuboids}")

                # Avoid accumulation of unused data in the GPU cache
                if epoch % 10 == 9:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA memory error occurred.")

        # Prepare the output at the end of the training and  return it
        best_model = torch.load(
            f"best_prior_cuda{self.hyperparams['device'][-1]}.pth"
        )
        print("Bestes Modell wieder geladen.")

        return (self.model.best_prior_config, self.model.best_prior_reconstruction, best_model) # Cuboid State Dictionary, Prior Rekonstruction, Model (CuboidPriorModel)
        
    def visualize_prior_reconstruction(
            self,
    ) -> None:
        """
        Visualization function for the prior reconstruction.
        """
        resulting_reconstruction = self.model.best_prior_reconstruction
        plot_3d_tensor_rel_plotly(
            resulting_reconstruction,
            filter=self.hyperparams['filter'],
            timesteps=self.hyperparams['timesteps'],
            space_range_xy=self.model.space_range_xy[1],
        ).write_html("best_prior_reconstruction_vis.html")


def new_cuboids_underreconstructed_cells(original_prior, current_prior, epoch_seed, device, epoch, nr_of_cuboids=5, new_half_size = 0.6):
    """ Function for adding new cuboids in under-reconstructed space-time cells. The twenty space-time cells with the largest 
  load deficit are determined, and new cuboids are added to the center of five of the cells (chosen at random), with a load value corresponding to the previous deficit. """
    pointwise_differences = original_prior - current_prior
    flattened_tensor = pointwise_differences.view(-1)

    topk_values, topk_indices = torch.topk(flattened_tensor, 20)
    topk_indices_3d = torch.unravel_index(topk_indices, pointwise_differences.shape)
    formatted_indices = list(zip(topk_indices_3d[0].tolist(), topk_indices_3d[1].tolist(), topk_indices_3d[2].tolist()))
    topk_values_and_indices = list(zip(topk_values.tolist(), formatted_indices))
    print(topk_values_and_indices)

    random.seed(epoch_seed)
    random_selection = random.sample(topk_values_and_indices, nr_of_cuboids)

    print(random_selection)

    random_selection_with_offset = [(value, (x + 0.5, y + 0.5, z + 0.5)) for value, (x, y, z) in random_selection]
    coordinates = [coords for _, coords in random_selection_with_offset]

    new_centers = torch.tensor(coordinates, dtype=torch.float32, device=device)

    values = [math.sqrt(value) for value, _ in random_selection_with_offset]

    new_radiations = torch.tensor(values, dtype=torch.float32, device=device)

    sqrt_value = math.sqrt(new_half_size)

    new_half_lengths_x = torch.full(
        (nr_of_cuboids,),
        sqrt_value,
        dtype=torch.float32,
        device=device,
    )

    new_half_widths_y = torch.full(
        (nr_of_cuboids,),
        sqrt_value,
        dtype=torch.float32,
        device=device,
    )

    new_half_heights_z = torch.full(
        (nr_of_cuboids,),
        sqrt_value,
        dtype=torch.float32,
        device=device,
    )

    return new_centers, new_radiations, new_half_lengths_x, new_half_widths_y, new_half_heights_z


def plot_3d_tensor_rel_plotly(
    tensor: torch.Tensor,
    filter: float,
    timesteps: int,
    space_range_xy: int,
) -> go.Figure:
    """
    Visualization function for the resulting 3D space-time reconstruction. Uses Plotly plot. Interactive 3D plot can be saved as an HTML file.
    """

    if tensor.requires_grad:
        tensor = tensor.detach()

    tensor_np = tensor.cpu().numpy()

    x, y, z = np.indices(tensor_np.shape)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = tensor_np.flatten()

    mask = values >= filter

    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    values_filtered = values[mask]

    hover_text = [
        f"({x}, {y}, {z}): {value:.4f}"
        for x, y, z, value in zip(x_filtered, y_filtered, z_filtered, values_filtered)
    ]

    fig = go.Figure(
        data=go.Scatter3d(
            x=x_filtered,
            y=y_filtered,
            z=z_filtered,
            mode="markers",
            marker=dict(
                size=5,
                color=values_filtered,
                colorscale="Viridis_r",
                opacity=0.8,
                colorbar=dict(title="Wertebereich"),
            ),
            text=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X AXIS",
            yaxis_title="Y AXIS",
            zaxis_title="Z AXIS",
            xaxis=dict(nticks=10, range=[0, space_range_xy]),
            yaxis=dict(nticks=10, range=[0, space_range_xy]),
            zaxis=dict(nticks=10, range=[0, timesteps]),
        ),
        width=700,
        margin=dict(r=10, b=10, l=10, t=10),
    )

    # Plot can be displayed with: fig.show()
    # Plot can be saved with: fig.write_html(‘name_of_3dplot.html’)

    return fig


def calculate_scen_0_rel_sum(training):
    """ Function for calculating the scenario error in percent (Rel. 0) of the cuboid prior and cuboid initialization with respect to the prior scenario (translation quality). """
    original_prior = training.load_prior_scenario().detach().cpu()
    original_sum = torch.sum(original_prior)
    pointwise_abs_differences = torch.abs(original_prior - training.model.best_prior_reconstruction.detach().cpu())
    differences_sum = torch.sum(pointwise_abs_differences)
    scen_0_rel_sum_result = (differences_sum / original_sum) * 100

    pointwise_abs_differences_init = torch.abs(original_prior - training.model.init_prior_reconstruction.detach().cpu())
    differences_sum_init = torch.sum(pointwise_abs_differences_init)
    scen_0_rel_sum_init = (differences_sum_init / original_sum) * 100
    print( f"Der prozentuale Szenariofehler beträgt {scen_0_rel_sum_result} %.")
    return scen_0_rel_sum_result, scen_0_rel_sum_init #Scenario error of the cuboid prior, scenario error of cuboid initialization (regarding prior scenario)