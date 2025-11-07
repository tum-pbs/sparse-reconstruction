from dataclasses import dataclass
from typing import Optional, Dict, Any

from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config, FrozenDict
from diffusers.utils import BaseOutput

import torch
from torch import nn
import pytorch_lightning as pl

import torch_geometric.data as geom_data
from torch_geometric.nn import MessagePassing, radius_graph, InstanceNorm
from torch_geometric.data import Data


def create_periodic_grid_graph(width: int, height: int):

    num_nodes = width * height
    nodes = torch.arange(num_nodes).reshape(height, width)

    edge_list = []
    for i in range(height):
        for j in range(width):
            node_idx = nodes[i, j].item()
            neighbors = [
                nodes[(i - 1) % height, j].item(),  # Up
                nodes[(i + 1) % height, j].item(),  # Down
                nodes[i, (j - 1) % width].item(),  # Left
                nodes[i, (j + 1) % width].item(),  # Right
            ]
            for neighbor_idx in neighbors:
                edge_list.append([node_idx, neighbor_idx])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)

    data = geom_data.Data(x=x, edge_index=edge_index)

    return data

class Swish(nn.Module):
    """
    Swish activation function
    """

    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + 2 + 3 + n_variables, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MPNN2DImpl(pl.LightningModule):
    def __init__(self, config):

        super().__init__()

        self.out_features = config.out_features
        self.hidden_features = config.hidden_features
        self.hidden_layer = config.hidden_layer
        self.time_window = config.time_window

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=1
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=1
                                         )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.out_features + 3, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        self.output_mlp = nn.Sequential(
            nn.Conv1d(1, 8, 16, stride=6),
            Swish(),
            nn.Conv1d(8, 1, 10, stride=1))

        self.project = nn.Linear(32, self.out_features)

    def forward(self, graph):

        u = graph.x
        pos = graph.pos
        batch = graph.batch

        edge_index = graph.edge_index

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos), -1)

        variables = torch.zeros(node_input.shape[0], 1).to(node_input.device)

        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos, variables, edge_index, batch)

        diff = self.output_mlp(h[:, None]).squeeze(1)
        diff = self.project(diff)

        out = u + diff

        return out


@dataclass
class MPNN2DOutput(BaseOutput):
    """
    The output of [`MPNN2D`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`
    """

    sample: "torch.Tensor"  # noqa: F821


class MPNN2D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            in_channels,
            width,
            height
    ):
        super().__init__()

        self.in_channels = in_channels
        self.width = width
        self.height = height

        config = {
            'out_features': in_channels,
            'hidden_features': 256,
            'hidden_layer': 6,
            'time_window': 1,
        }
        config = FrozenDict(config)

        edge_index = (
            create_periodic_grid_graph(self.width, self.height).edge_index)

        self.register_buffer('edge_index', edge_index, persistent=False)

        self.model = MPNN2DImpl(config)

    def build_graph(self, data):

        u = data.reshape(-1, self.in_channels)

        assert data.size(0) == self.height
        assert data.size(1) == self.width

        pos_x = torch.linspace(0, 1, self.height).float().to(data.device)
        pos_y = torch.linspace(0, 1, self.width).float().to(data.device)

        pos_x, pos_y = torch.meshgrid(pos_x, pos_y)

        pos_x = pos_x.reshape(-1, 1)
        pos_y = pos_y.reshape(-1, 1)

        pos_t = torch.zeros_like(pos_x)

        graph = Data(x=u, edge_index=self.edge_index)

        graph.pos = torch.cat([pos_x, pos_y, pos_t], 1)

        graph.batch = torch.tensor([1]).to(data.device)

        return graph

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):

        assert hidden_states.dim() == 4, "Expected hidden_states to have 4 dimensions (batch_size, num_channels, height, width)"

        assert hidden_states.size(0) == 1, "Use MPNN2D for batch size 1 only."

        hidden_states = hidden_states[0].permute(1, 2, 0)

        height = hidden_states.size(0)
        width = hidden_states.size(1)

        graph = self.build_graph(hidden_states)

        output = self.model(graph)

        output = output.reshape(height, width, -1).unsqueeze(0)
        output = output.permute(0, 3, 1, 2)

        if not return_dict:
            return (output,)

        return MPNN2DOutput(sample=output)