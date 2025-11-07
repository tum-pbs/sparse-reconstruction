import torch
import torch.nn as nn
import torch.functional as F


def one_hot_encoding(size_encoding: int, index_tensor: torch.Tensor) -> torch.Tensor:
    '''
    One hot encoding of a tensor

    Args:
        size_encoding: length of the encoding
        index_tensor: tensor to be encoded with shape [batch_size, 1]
    Return:
        one hot encoded tensor with shape [batch_size, size_encoding]
    '''
    assert torch.max(index_tensor) < size_encoding, "Index tensor contains too large values! Increase encoding size."

    enc = torch.zeros(index_tensor.shape[0], size_encoding, device=index_tensor.device)
    enc = torch.scatter(enc, 1, index_tensor.to(torch.int64), 1)
    return enc


def stacked_one_hot_encoding(size_encoding: int, index_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Multiple stacked (smaller) one hot encodings sized size_encoding//index_length for a tensor with multiple indices
    (additionally pads the encoding to the full size).

    Args:
        size_encoding: length of the encoding
        index_tensor: tensor to be encoded with shape [batch_size, index_length]
    Return:
        stacked one hot encoded tensor with shape [batch_size, size_encoding]
    '''
    reduced_size = size_encoding // index_tensor.shape[1]

    assert torch.max(index_tensor) < reduced_size, "Index tensor contains too large values! Increase encoding size."

    enc = one_hot_encoding(reduced_size, index_tensor.flatten().unsqueeze(1))
    enc = enc.view(index_tensor.shape[0], -1)

    pad = torch.zeros(index_tensor.shape[0], size_encoding - enc.shape[1], device=index_tensor.device)
    enc = torch.cat([enc, pad], dim=1)

    return enc


def multi_hot_encoding(size_encoding: int, index_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Multi hot encoding of a tensor (several indices are set to one)

    Args:
        size_encoding: length of the encoding
        index_tensor: tensor to be encoded with shape [batch_size, index_length]
    Return:
        multi hot encoded tensor with shape [batch_size, size_encoding]
    '''
    assert torch.max(index_tensor) < size_encoding, "Index tensor contains too large values! Increase encoding size."

    # Slightly hacky solution to filter out negative placeholder values
    index_tensor = index_tensor.clone()
    index_tensor[index_tensor < 0] = size_encoding  # Temporarily set to an out-of-bounds index

    enc = torch.zeros(index_tensor.shape[0], size_encoding + 1, device=index_tensor.device)  # Add extra column for out-of-bounds
    enc = torch.scatter(enc, 1, index_tensor.to(torch.int64), 1)
    enc = enc[:, :-1]  # Remove the extra column
    return enc


def mlp(in_channels:int, out_channels:int, num_hidden:int, size_hidden:int, dropout:float) -> nn.Module:
    '''
    Create a MLP (FC+ReLU) with the given number of layers

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        num_hidden: number of layers
        size_hidden: size of the hidden layers
        dropout: dropout rate
    Return:
        MLP with the given number of layers
    '''
    if num_hidden == 0:
        return nn.Linear(in_channels, out_channels)

    layers = [
        nn.Dropout(dropout),
        nn.Linear(in_channels, size_hidden),
        nn.ReLU(),
    ]
    for _ in range(num_hidden-1):
        layers+= [
            nn.Linear(size_hidden, size_hidden),
            nn.ReLU(),
        ]
    layers += [
        nn.Linear(size_hidden, out_channels),
    ]

    return nn.Sequential(*layers)

def pad_constants(constants: torch.Tensor, size: int) -> torch.Tensor:
    '''
    Pad the constants tensor to the given size

    Args:
        constants: tensor with the constants of shape [batch_size, num_constants]
        size: target size of constants dimension
    Return:
        padded tensor with shape [batch_size, size]
    '''
    pad = torch.zeros(constants.shape[0], size - constants.shape[1], device=constants.device)
    return torch.cat([constants, pad], dim=1)



class TaskEmbedding(nn.Module):
    def __init__(self, size_hot_encoding:int = 64, num_hidden:int = 3, size_hidden:int= 64, dropout:float=0.05, size_embedding:int = 512):
        '''
        Task embedding module for the metadata added with the metadata dataset. It contains five separate embedding
        MLPs for the different metadata.
        1. PDE: Value from all available PDEs with empty space for new PDEs -> one-hot encoding -> MLP
        2. Fields: Values from all available fields with empty space for new fields -> multi-hot encoding -> MLP
        3. Constants: Values from all available constants with empty space for new constants -> multi-hot encoding -> MLP
        4. Boundary conditions: Values from all available boundary conditions with empty space for new boundary conditions -> stacked one-hot encoding -> MLP
        5. Scalar physical metadata: domain extent, dt, reynolds number, varied constant values for each simulation (+padding) -> MLP

        Then, all 5 embeddings are concatenated and passed through another MLP to get the final task embedding.

        Args:
            size_hot_encoding: size of the one-hot/stacked one-hot/multi-hot encodings. Number 5. gets padded to this size
            num_hidden: number of hidden layers in the MLPs
            size_hidden: size of the hidden layers in the MLPs
            dropout: dropout rate for the first layer in the MLPs
            size_embedding: size of the final task embedding
        '''
        super().__init__()
        self.size_hot_encoding = size_hot_encoding
        self.size_embedding = size_embedding

        self.pde_embedding      = mlp(size_hot_encoding, size_hidden, num_hidden, size_hidden, dropout)
        self.field_embedding    = mlp(size_hot_encoding, size_hidden, num_hidden, size_hidden, dropout)
        self.constant_embedding = mlp(size_hot_encoding, size_hidden, num_hidden, size_hidden, dropout)
        self.boundary_embedding = mlp(size_hot_encoding, size_hidden, num_hidden, size_hidden, dropout)
        self.physical_embedding = mlp(size_hot_encoding, size_hidden, num_hidden, size_hidden, dropout)

        self.final_embedding = mlp(5 * size_hidden, size_embedding, num_hidden=1, size_hidden=5*size_hidden, dropout=0)


    def forward(self, sample):
        data = sample["data"]
        constants_norm = sample["constants_norm"]
        constants_raw = sample["constants"]
        time_step_stride = sample["time_step_stride"]
        metadata = sample["physical_metadata"]
        loading_metadata = sample["loading_metadata"]

        hot_pde = one_hot_encoding(self.size_hot_encoding, metadata["PDE"])
        emb_pde = self.pde_embedding(hot_pde)

        hot_fields = multi_hot_encoding(self.size_hot_encoding, metadata["Fields"])
        emb_fields = self.field_embedding(hot_fields)

        hot_boundary = stacked_one_hot_encoding(self.size_hot_encoding, metadata["Boundary Conditions"])
        emb_boundary = self.boundary_embedding(hot_boundary)

        hot_constants = multi_hot_encoding(self.size_hot_encoding, metadata["Constants"])
        emb_constants = self.constant_embedding(hot_constants)

        physical = torch.cat([
            metadata["Domain Extent"],
            metadata["Dt"] * time_step_stride,
            metadata["Reynolds Number"],
            constants_norm,
        ], dim=1)
        physical = pad_constants(physical, self.size_hot_encoding)
        emb_physical = self.physical_embedding(physical)

        embedding = torch.cat([emb_pde, emb_fields, emb_constants, emb_boundary, emb_physical], dim=1)
        return self.final_embedding(embedding)

