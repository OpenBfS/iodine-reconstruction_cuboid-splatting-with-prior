import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import kmeans_plusplus
from scipy.spatial import KDTree
import torch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Cuboid Initialization

class CuboidSfmInit():

    def __init__(
            self,
            nr_of_cuboids,
            space_range_xy,
            space_range_z,
            seed,
            prior_data_directory: str,
            threshold,
            k_nearest_neighbors,
    ):
        """
        Constructor that creates an object of the CuboidSfmInit class and defines or initializes all attributes relevant to the point cloud initialization process.
        """
        
        self.nr_of_cuboids = nr_of_cuboids
        self.space_range_xy = space_range_xy
        self.space_range_z = space_range_z
        self.seed = seed
        self.k_nearest_neighbors = k_nearest_neighbors
        self.prior_scenario = self.load_prior(prior_data_directory)
        self.df_4d = self.create_4d_dataset(self.prior_scenario, threshold=threshold) 

        self.df_centers_radiations_distances = pd.DataFrame()
        self.cuboid_state_dict = {}
        self.adapted_cuboid_state_dict = {}

    def load_prior(self, prior_data_directory):
        """ Loads a prior scenario from the specified prior_data_directory."""
        prior_scenario = np.load(prior_data_directory)
        return prior_scenario
    
    def create_4d_dataset(self, prior_scenario: np.array, threshold: float) -> pd.DataFrame:

        """ 
        Generates a 4D data set from x, y, z, and iodine value from a 3D prior scenario. 
        prior_scenario: Prior scenario to be converted into 4D data points or data set.
        threshold: Threshold value above which load values must lie in order to be included in the data set of points, e.g., 0, 0.0, or 1.0.
        """

        x, y, z = np.indices(prior_scenario.shape)

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        values_flat = prior_scenario.flatten()

        dataset = np.column_stack((x_flat, y_flat, z_flat, values_flat))
        df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'value'])

        df_thresh = df[df["value"] > threshold].copy()

        return df_thresh
    
    def run(self):
        """ Function that performs the entire point cloud initialization process. 
        Saves the completed cuboid state dictionary from cuboid initialization in the adapted_cuboid_state_dict attribute. """
        
        df_centers_radiations = self.select_centers_from_dataset(self.df_4d, self.nr_of_cuboids, self.seed)
        self.df_centers_radiations_distances = self.get_avg_distances_to_closest_centers(df_centers_radiations, self.k_nearest_neighbors)
        self.cuboid_state_dict = self.create_cuboid_state_dict(self.df_centers_radiations_distances)
        self.adapted_cuboid_state_dict = self.adapt_inputs_to_supervised_approach(self.cuboid_state_dict)

    def select_centers_from_dataset(self, df_thresh, nr_of_cuboids, seed) -> pd.DataFrame: 
        """
        Uses k-means++ initialization to select suitable centers for the cuboids from the points in the data set.
        Creates a data set of the new cuboids from these points and their associated iodine values and returns it.
        """
        points_array = df_thresh[['x', 'y', 'z']].values 

        centers, _ = kmeans_plusplus(points_array, n_clusters=nr_of_cuboids, random_state=seed)

        centers_df = pd.DataFrame(centers, columns=['x', 'y', 'z'])

        merged_df = pd.merge(centers_df, df_thresh, on=['x', 'y', 'z'], how='left')

        merged_df[['x', 'y', 'z']] += 0.5

        return merged_df # Centers and iodine values of the new cuboids
    
    def get_avg_distances_to_closest_centers(self, merged_df, k=3):
        """ 
        Calculates the average distance to the k nearest centers. 
        For each point, the k nearest neighbors are determined using KDTree. 
        In the code, k+1, since the nearest point is the point itself. 
        The distance to the k neighbors is determined and the average is calculated. 
        This average is added to the data set of the new cuboids.
        """

        points = merged_df[['x', 'y', 'z']].values

        tree = KDTree(points)

        k=k

        distances, indices = tree.query(points, k=k+1)

        average_distances = distances[:, 1:].mean(axis=1)

        merged_df['average_distance'] = average_distances

        return merged_df
    
    def create_cuboid_state_dict(self, merged_df) -> dict:
        """ Generates the cuboid state dictionary for the generated point cloud initialization. """

        df_cub = merged_df.copy()
        df_cub["centers"] = df_cub.apply(lambda row: [row["x"], row["y"], row["z"]], axis=1)
        df_cub["half_lengths_x"] = df_cub["average_distance"] 
        df_cub["half_widths_y"] = df_cub["average_distance"] 
        df_cub["half_heights_z"] = df_cub["average_distance"] 
        df_cub.drop(columns=["x", "y", "z", "average_distance"], inplace=True)
        df_cub.rename(columns={"value":"radiations"}, inplace=True)

        tensor_dict = {col: torch.tensor(df_cub[col].tolist(), dtype=torch.float32) if col == "centers" else torch.tensor(df_cub[col].values, dtype=torch.float32) for col in df_cub.columns}

        return tensor_dict
    
    def adapt_inputs_to_supervised_approach(self, cuboid_state_dict):
        """ 
        Converts values in the Cuboid State Dictionary into square roots of iodine values and shape parameters, 
        as these are squared in supervised pretraining and all fine-tuning approaches 
        to avoid negative values in the calculations.
        """

        cuboid_state_dict = cuboid_state_dict.copy()
        cuboid_state_dict["half_lengths_x"] = torch.sqrt(cuboid_state_dict["half_lengths_x"])
        cuboid_state_dict["half_widths_y"] = torch.sqrt(cuboid_state_dict["half_widths_y"])
        cuboid_state_dict["half_heights_z"] = torch.sqrt(cuboid_state_dict["half_heights_z"])
        cuboid_state_dict["radiations"] = torch.sqrt(cuboid_state_dict["radiations"])

        return cuboid_state_dict
    
    def plot_cuboids_in_3d_space(self, divisor = 10) -> None:
        """
        Function that plots the current cuboids in 3D space. This is done based on their positions in space and their lengths, widths, and heights.
        The transparency reflects the iodine value. The divisor parameter scales the transparency. Example: divisor = 10 -> transparent if load is between 0 and 10, >10 not transparent.
        """
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")

        cuboid_state_dict = self.cuboid_state_dict
        space_range_xy = self.space_range_xy
        space_range_z = self.space_range_z

        for i in range(cuboid_state_dict["centers"].size(0)):
            center = cuboid_state_dict["centers"][i].detach().cpu().numpy()
            half_length_x = (cuboid_state_dict["half_lengths_x"][i]).detach().cpu().item()
            half_width_y = (cuboid_state_dict["half_widths_y"][i]).detach().cpu().item()
            half_height_z = (cuboid_state_dict["half_heights_z"][i]).detach().cpu().item()
            opacity = ((cuboid_state_dict["radiations"][i])/divisor).detach().cpu().item()

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
                np.ptp(space_range_xy),
                np.ptp(space_range_xy),
                np.ptp(space_range_z),
            ]
        )

        ax.set_xlim(space_range_xy)
        ax.set_ylim(space_range_xy)
        ax.set_zlim(space_range_z)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()
