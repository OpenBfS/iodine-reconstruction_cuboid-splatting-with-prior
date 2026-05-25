import torch
import json

from typing import Any

import plotly.graph_objects as go
import numpy as np


def get_true_exposures(paths_data_list: list[dict[str, Any]]) -> torch.Tensor:
    """
    Helper function to output the actual exposure values of the paths.
    """
    true_exposures = [path_data["exposure"] for path_data in paths_data_list]
    true_exposures_tensor = torch.tensor(true_exposures, dtype=torch.float32)
    return true_exposures_tensor


def get_calculated_exposures(
    paths_data_list: list[dict[str, Any]],
    model 
) -> torch.Tensor:
    """
    Helper function to output the exposure values of the paths calculated according to the current state of the model (the current reconstruction).
    """
    with torch.no_grad():
        calculated_exposures = model(paths_data_list)
    return torch.as_tensor(calculated_exposures)


def load_cuboid_state_dict(path: str) -> dict:
    """ Function for loading a cuboid state dictionary. """
    with open(path, 'r', encoding='utf-8') as file:
        cuboid_state_dict = json.load(file)

    for key, value in cuboid_state_dict.items():
        cuboid_state_dict[key] = torch.tensor(value, dtype=torch.float32)

    return cuboid_state_dict


def plot_3d_tensor_rel_plotly(
    tensor: torch.Tensor,
    filter: float,
    timesteps: int,
    space_range_xy: int,
) -> go.Figure:
    """
    Visualization function for the resulting 3D space-time reconstruction. 
    Uses Plotly plot. Interactive 3D plot can be saved as an HTML file.
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