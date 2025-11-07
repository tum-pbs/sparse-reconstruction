import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import random
from scipy.interpolate import griddata
from typing import Optional
import ot as pot
from functools import partial
import math
import torch.nn as nn
from LSIM_3D.src.volsim.distance_model import *
from torchfsm.operator import Div, Leray
from torchfsm.mesh import MeshGrid
from torchfsm.operator import Grad
from torchfsm.functional import curl
from torchfsm.plot import plot_field
from scipy.ndimage import zoom
from scipy.ndimage import laplace
import h5py
from vape4d import render
from copy import deepcopy
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
import pyvista as pv
from scipy.stats import gaussian_kde
import torch.nn.functional as F
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import NullLocator


def diverging_alpha(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
        cmap.colors = deepcopy(cmap.colors)
        for i, a in enumerate(cmap.colors):
            a.append(2 * abs(i / cmap.N - 0.5))
    elif isinstance(cmap, LinearSegmentedColormap):
        cmap._segmentdata["alpha"] = np.array(
            [[0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )
    return cmap

def m_shape_alpha(cmap):
    """Adds M-shaped alpha to a colormap (opaque at 0.25 & 0.75, transparent at 0.0, 0.5, 1.0)"""
    cmap = cmap.copy()
    tmin, tmax = (0.49, 0.51)

    if isinstance(cmap, ListedColormap):
        cmap.colors = [list(rgba) for rgba in cmap.colors]
        N = cmap.N
        for i, color in enumerate(cmap.colors):
            x = i / (N - 1)
            # Fully transparent in the flat range
            if tmin <= x <= tmax:
                alpha = 0.0
            elif x < tmin:
                # left side: peak at 0.25
                alpha = max(0, 1 - abs((x - 0.25) / (tmin - 0.25)))
            else:
                # right side: peak at 0.75
                alpha = max(0, 1 - abs((x - 0.75) / (0.75 - tmax)))
            color[3] = min(1, max(0, alpha))
    elif isinstance(cmap, LinearSegmentedColormap):
        # You can define this segmentally too, but linear is smoother with ListedColormap
        raise NotImplementedError("Use ListedColormap for fine control.")
    else:
        raise TypeError("Unsupported colormap type")

    return cmap

def step_alpha(cmap, alpha_start=0.2, alpha_end=0.8):
    """
    Applies a step-shaped alpha mask to a colormap:
    - Transparent outside [alpha_start, alpha_end]
    - Fully opaque within [alpha_start, alpha_end]
    
    Parameters:
        cmap: ListedColormap
        alpha_start: float in [0, 1], start of the opaque region
        alpha_end: float in [0, 1], end of the opaque region
    """
    cmap = cmap.copy()
    
    if isinstance(cmap, ListedColormap):
        cmap.colors = [list(rgba) for rgba in cmap.colors]
        N = cmap.N
        for i, color in enumerate(cmap.colors):
            x = i / (N - 1)
            # Step shape: transparent before and after range, opaque inside
            alpha = 1.0 if alpha_start <= x <= alpha_end else 0.0
            color[3] = alpha
    else:
        raise TypeError("Only ListedColormap is supported for step alpha.")
    
    return cmap

def ramp_step_alpha(cmap, ramp_end=0.2, alpha_cutoff=0.8):
    """
    Applies a ramp-to-step alpha profile to a ListedColormap.
    
    Alpha profile:
    - 0.0 at x = 0.0
    - Linearly ramps to 1.0 at x = ramp_end
    - Remains at 1.0 until x = alpha_cutoff
    - Drops to 0.0 after x = alpha_cutoff

    Parameters:
        cmap: ListedColormap
        ramp_end: float in (0, 1), end of linear ramp-up
        alpha_cutoff: float in (ramp_end, 1), where the alpha drops back to 0
    """
    cmap = cmap.copy()

    if isinstance(cmap, ListedColormap):
        cmap.colors = [list(rgba) for rgba in cmap.colors]
        N = cmap.N
        for i, color in enumerate(cmap.colors):
            x = i / (N - 1)
            if x < ramp_end:
                alpha = x / ramp_end  # linear ramp-up
            elif x < alpha_cutoff:
                alpha = 1.0           # constant max
            else:
                alpha = 0.0           # hard drop
            color[3] = alpha
    else:
        raise TypeError("Only ListedColormap is supported.")

    return cmap

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape).to(x.device)
        std = self.std.reshape(*shape).to(x.device)
        return (x - mean) / std

    def inverse(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape).to(x.device)
        std = self.std.reshape(*shape).to(x.device)
        return x * std + mean

    def scale(self):
        return self.std
    
def compute_statistics(data):
    mean = data.mean(axis=(0,2,3,4))
    std = data.std(axis=(0,2,3,4))
    return mean, std

def spectral_resize_3d(img, target_size):
    # img: (D, H, W), target_size: int
    original_shape = np.array(img.shape)
    scale_factor = np.prod(original_shape) / (target_size ** 3)  # normalize energy

    F = fftn(img)
    F_shifted = fftshift(F)
    center = original_shape // 2
    half_size = target_size // 2

    # Crop the central part of the spectrum
    cropped = F_shifted[
        center[0] - half_size:center[0] + half_size,
        center[1] - half_size:center[1] + half_size,
        center[2] - half_size:center[2] + half_size
    ]

    cropped_unshifted = ifftshift(cropped)
    resized = ifftn(cropped_unshifted)
    resized = np.real(resized) / scale_factor  # apply normalization

    return resized
    
def plot_slice(data, snapshot_idx, channel_idx, slice_idx, name=None, direction='z'):
    """
    Plot a slice of the data at a specific snapshot and channel.
    
    Parameters:
    - data: The dataset containing the flow field.
    - snapshot_idx: Index of the time snapshot to plot.
    - channel_idx: Index of the channel to plot.
    - slice_idx: Index of the slice along the z-axis.
    """
    # Extract the specific snapshot, channel, and slice
    snapshot = data[snapshot_idx]  # Shape: (channels, Nx, Ny, Nz)
    channel_data = snapshot[channel_idx]  # Shape: (Nx, Ny, Nz)
    
    if direction == 'x':
        slice_data = channel_data[slice_idx, :, :]  # Shape: (Nx, Ny)
    if direction == 'y':
        slice_data = channel_data[:, slice_idx, :]  # Shape: (Nx, Ny)
    if direction == 'z':
        slice_data = channel_data[:, :, slice_idx]  # Shape: (Nx, Ny)

    # Plot the selected slice
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, cmap='twilight', origin='lower')
    plt.colorbar()
    plt.title(f'Snapshot {snapshot_idx}, Channel {channel_idx}, Slice {slice_idx}', fontsize=18)
    plt.xlabel('X-axis', fontsize=18)
    plt.ylabel('Y-axis', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save the plot instead of showing it
    if name is None:
        output_file = f'generated_plots/snapshot_{snapshot_idx}_channel_{channel_idx}_slice_{slice_idx}.png'
    else:
        output_file = f'generated_plots/{name}.png'
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f'Plot saved as {output_file}')
    
def plot_slice_together(data, snapshot_idx, slice_idx, name=None, direction='z'):
    # Extract the specific snapshot, channel, and slice
    snapshot = data[snapshot_idx]  # Shape: (channels, Nx, Ny, Nz)
    vx = snapshot[0]  # Shape: (Nx, Ny, Nz)
    vy = snapshot[1]  # Shape: (Nx, Ny, Nz)
    vz = snapshot[2]  # Shape: (Nx, Ny, Nz)
    
    if direction == 'x':
        vx_slice = vx[slice_idx, :, :]  # Shape: (Nx, Ny)
        vy_slice = vy[slice_idx, :, :]  # Shape: (Nx, Ny)
        vz_slice = vz[slice_idx, :, :]  # Shape: (Nx, Ny)
    if direction == 'y':
        vx_slice = vx[:, slice_idx, :]  # Shape: (Nx, Ny)
        vy_slice = vy[:, slice_idx, :]  # Shape: (Nx, Ny)
        vz_slice = vz[:, slice_idx, :]  # Shape: (Nx, Ny)
    if direction == 'z':
        vx_slice = vx[:, :, slice_idx]  # Shape: (Nx, Ny)
        vy_slice = vy[:, :, slice_idx]  # Shape: (Nx, Ny)
        vz_slice = vz[:, :, slice_idx]  # Shape: (Nx, Ny)
        
    # Find global min and max for colorbar
    vmin = min(vx_slice.min(), vy_slice.min(), vz_slice.min())
    vmax = max(vx_slice.max(), vy_slice.max(), vz_slice.max())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
    im0 = axes[0].imshow(vx_slice, cmap='twilight', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title("x-velocity", fontsize=22)
    axes[0].axis('off')
    im1 = axes[1].imshow(vy_slice, cmap='twilight', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("y-velocity", fontsize=22)
    axes[1].axis('off')
    im2 = axes[2].imshow(vz_slice, cmap='twilight', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title("z-velocity", fontsize=22)
    axes[2].axis('off')
    
    axes[3].axis('off')
    cbar = fig.colorbar(im2, ax=axes[3], fraction=1.0)
    cbar.ax.tick_params(labelsize=18)
    
    if name is None:
        output_file = f'generated_plots/snapshot_{snapshot_idx}_slice_{slice_idx}_together.png'
    else:
        output_file = f'generated_plots/{name}.png'
        
    plt.subplots_adjust(wspace=0.05, left=0.05, right=0.98, top=0.92, bottom=0.08)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Plot saved as {output_file}')
    
    
def plot_2d_comparison(low_res, high_res, gt, filename):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.title("Low-Resolution Input")
    plt.imshow(low_res, cmap="twilight")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Super-Resolved Output")
    plt.imshow(high_res, cmap="twilight")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground truth")
    plt.imshow(gt, cmap="twilight")
    plt.axis('off')

    # Save the plot as a PNG file
    plt.tight_layout()
    filename = f"generated_plots/{filename}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved in {filename}")
    
def plot_report(dataset, config, indices, samples_x, samples_y, recons, filename):
    n = len(indices)
    #fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(16, 4 * n))
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(10, 4 * n))

    for row, idx in enumerate(indices):
        vmin = min(
            samples_x[idx].min(),
            recons[idx].min(),
            samples_y[idx].min()
        )
        vmax = max(
            samples_x[idx].max(),
            recons[idx].max(),
            samples_y[idx].max()
        )
        
        # Low-res
        axes[row, 0].imshow(
            samples_x[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            vmin=vmin,
            vmax=vmax
        )
        axes[row, 0].axis('off')
        
        # Super-res
        axes[row, 1].imshow(
            recons[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            vmin=vmin,
            vmax=vmax
        )
        axes[row, 1].axis('off')
        
        # Ground truth
        im = axes[row, 2].imshow(
            samples_y[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            vmin=vmin,
            vmax=vmax
        )
        axes[row, 2].axis('off')
        
        # Leave the fourth subplot blank but place the colorbar there
        axes[row, 3].axis('off')
        cbar = fig.colorbar(im, ax=axes[row, 3], fraction=1.0)
        cbar.ax.tick_params(labelsize=10)

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation")
    axes[0, 1].set_title("Reconstruction")
    axes[0, 2].set_title("Reference")

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    
def plot_report_hybrid(dataset, config, indices, samples_x, samples_y, recons, first_step, filename):
    n = len(indices)
    #fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(16, 4 * n))
    fig, axes = plt.subplots(nrows=n, ncols=5, figsize=(14, 4 * n))

    for row, idx in enumerate(indices):
        vmin = min(
            samples_x[idx].min(),
            recons[idx].min(),
            samples_y[idx].min(),
            first_step[idx].min()
        )
        vmax = max(
            samples_x[idx].max(),
            recons[idx].max(),
            samples_y[idx].max(),
            first_step[idx].max()
        )
        
        # Low-res
        axes[row, 0].imshow(
            samples_x[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            #vmin=vmin,
            #vmax=vmax
        )
        axes[row, 0].axis('off')
        
        # 1st step
        axes[row, 1].imshow(
            first_step[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            #vmin=vmin,
            #vmax=vmax
        )
        axes[row, 1].axis('off')
        
        # Super-res
        axes[row, 2].imshow(
            recons[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            #vmin=vmin,
            #vmax=vmax
        )
        axes[row, 2].axis('off')
        
        # Ground truth
        im = axes[row, 3].imshow(
            samples_y[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
            #vmin=vmin,
            #vmax=vmax
        )
        axes[row, 3].axis('off')
        
        # Leave the fourth subplot blank but place the colorbar there
        axes[row, 4].axis('off')
        cbar = fig.colorbar(im, ax=axes[row, 4], fraction=1.0)
        cbar.ax.tick_params(labelsize=10)

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation")
    axes[0, 1].set_title("1st step")
    axes[0, 2].set_title("Reconstruction")
    axes[0, 3].set_title("Reference")

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    
def plot_report_3d(dataset, config, indices, samples_x, samples_y, recons, filename):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(16, 4 * n))
    
    base_cmap = plt.get_cmap("turbo")
    listed_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    colormap = ramp_step_alpha(listed_cmap, 0.2, 0.99)

    for row, idx in enumerate(indices):
        # Low-res
        velocity = samples_x[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.detach().cpu().numpy()
        img0 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0, 
        )
        axes[row, 0].imshow(img0)
        axes[row, 0].axis("off")
        
        # Super-res
        velocity = recons[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.detach().cpu().numpy()
        img1 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 1].imshow(img1)
        axes[row, 1].axis("off")
        
        # Ground truth
        velocity = samples_y[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.detach().cpu().numpy()
        img2 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 2].imshow(img2)
        axes[row, 2].axis("off")

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation")
    axes[0, 1].set_title("Reconstruction")
    axes[0, 2].set_title("Reference")

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    

def plot_report_3d_iso(dataset, config, indices, samples_x, samples_y, recons, filename):
    os.makedirs("generated_plots/frames", exist_ok=True)

    qs_interp = []
    qs_recons = []
    qs_ref = []

    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(16, 4 * n))
    
    base_cmap = plt.get_cmap("turbo")
    listed_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    colormap = ramp_step_alpha(listed_cmap, 0.2, 0.99)

    for row, idx in enumerate(indices):
        for col, (data, title) in enumerate([
            (samples_x[idx], "Interpolation"),
            (recons[idx], "Reconstruction"),
            (samples_y[idx], "Reference"),
        ]):
            velocity = data.unsqueeze(0)
            # Compute Q tensor for isosurface plotting
            if dataset is not None:
                q = q_criterion(velocity, 128)
            else:
                q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
                
            q_flat = q[0, 0].detach().cpu().numpy().flatten()
            if col == 0:
                qs_interp.append(q_flat)
            if col == 1:
                qs_recons.append(q_flat)
            if col == 2:
                qs_ref.append(q_flat)
            
            save_path = f"generated_plots/frames/frame_{filename}_row{row}_col{col}.png"
            
            # Call your isosurface plot function here:
            if dataset is not None:
                plot_q_isosurface(
                    Q_tensor=q,
                    velocity=velocity,
                    q_threshold=150,
                    spacing=(2*np.pi/128, 2*np.pi/128, 2*np.pi/128),
                    cmap="jet",
                    save_path=save_path,
                )
            else:
                plot_q_isosurface_ch(
                    Q_tensor=q,
                    velocity=velocity,
                    vort_threshold=20,
                    q_threshold=5,
                    spacing=(2*np.pi/192, 1/65, np.pi/192),
                    cmap="jet",
                    save_path=save_path,
                )
                
            
            img = plt.imread(save_path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            
            if row == 0:
                axes[row, col].set_title(title)

    plt.tight_layout()
    final_path = f"generated_plots/{filename}.png"
    plt.savefig(final_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in {final_path}")
    
    method_qs = {
        "Interpolation": np.concatenate(qs_interp),
        "Reconstruction": np.concatenate(qs_recons),
        "Reference": np.concatenate(qs_ref),
    }

    plt.figure(figsize=(8, 5))

    # Define x-range for evaluation
    x_values = np.linspace(-100, 100, 1000)

    for label, q_values in method_qs.items():
        q_pos = q_values
        kde = gaussian_kde(q_pos)
        density = kde(x_values)
        plt.plot(x_values, density, label=label, linewidth=2)

    plt.xlabel("Q value")
    plt.ylabel("Probability Density")
    plt.title("Q-criterion Distribution per Method")
    plt.legend()
    plt.grid(True)

    dist_path = f"generated_plots/q_distribution_kde_{filename}.png"
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Q-distribution KDE plot saved in {dist_path}")


    
def plot_report_5(dataset, config, indices, samples_x, samples_y, recons, reg, shu, filename):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=6, figsize=(18, 4 * n))

    for row, idx in enumerate(indices):
        
        # Low-res
        axes[row, 0].imshow(
            samples_x[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 0].axis('off')
        
        # Regression
        axes[row, 1].imshow(
            reg[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 1].axis('off')
        
        # Shu
        axes[row, 2].imshow(
            shu[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 2].axis('off')
        
        # Super-res
        axes[row, 3].imshow(
            recons[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 3].axis('off')
        
        # Ground truth
        im = axes[row, 4].imshow(
            samples_y[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 4].axis('off')
        
        # Leave the fourth subplot blank but place the colorbar there
        axes[row, 5].axis('off')
        cbar = fig.colorbar(im, ax=axes[row, 5], fraction=1.0)
        cbar.ax.tick_params(labelsize=10)

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation", fontsize=20)
    axes[0, 1].set_title("Supervised baseline", fontsize=20)
    axes[0, 2].set_title("Shu et al. method", fontsize=20)
    axes[0, 3].set_title("Masked diffusion (ours)", fontsize=20)
    axes[0, 4].set_title("Reference", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    
    compare_spectrum_4(reg, shu, recons, samples_y, filename)
    
def plot_report_5_3d(dataset, config, indices, samples_x, samples_y, recons, reg, shu, filename):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=5, figsize=(18, 4 * n))
    
    base_cmap = plt.get_cmap("turbo")
    listed_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    colormap = ramp_step_alpha(listed_cmap, 0.5, 0.99)

    for row, idx in enumerate(indices):
        # Low-res
        velocity = samples_x[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img0 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 0].imshow(img0)
        axes[row, 0].axis("off")
        
        # Regression
        velocity = reg[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img1 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 1].imshow(img1)
        axes[row, 1].axis("off")
        
        # Shu
        velocity = shu[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img2 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 2].imshow(img2)
        axes[row, 2].axis("off")
        
        # Super-res
        velocity = recons[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img3 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 3].imshow(img3)
        axes[row, 3].axis("off")
        
        # Ground truth
        velocity = samples_y[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img4 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 4].imshow(img4)
        axes[row, 4].axis("off")

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation")
    axes[0, 1].set_title("P3D baseline")
    axes[0, 2].set_title("Shu method")
    axes[0, 3].set_title("Mask method")
    axes[0, 4].set_title("Reference")

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    
def plot_report_5_3d_iso(dataset, config, indices, samples_x, samples_y, recons, reg, shu, filename):
    os.makedirs("generated_plots/frames", exist_ok=True)
    
    qs_interp = []
    qs_base = []
    qs_shu = []
    qs_diff = []
    qs_ref = []
    
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=5, figsize=(14, 3 * n))

    for row, idx in enumerate(indices):
        entries = [
            ("Interpolation", samples_x[idx]),
            ("Supervised baseline", reg[idx]),
            ("Shu et al. method", shu[idx]),
            ("Masked diffusion (ours)", recons[idx]),
            ("Reference", samples_y[idx]),
        ]
        
        for col, (title, velocity_tensor) in enumerate(entries):
            velocity = velocity_tensor.unsqueeze(0)
            if dataset is not None:
                q = q_criterion(velocity, 128)
            else:
                q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
            
            q_flat = q[0, 0].detach().cpu().numpy().flatten()
            
            if col == 0:
                qs_interp.append(q_flat)
            if col == 1:
                qs_base.append(q_flat)
            if col == 2:
                qs_shu.append(q_flat)
            if col == 3:
                qs_diff.append(q_flat)
            if col == 4:
                qs_ref.append(q_flat)

            save_path = f"generated_plots/frames/frame_{filename}_row{row}_col{col}.png"
            if dataset is not None:
                plot_q_isosurface(
                    Q_tensor=q,
                    velocity=velocity,
                    q_threshold=60,
                    vort_threshold=35,
                    spacing=(2*np.pi/128, 2*np.pi/128, 2*np.pi/128),
                    cmap="jet",
                    save_path=save_path,
                )
            else:
                plot_q_isosurface_ch(
                    Q_tensor=q,
                    velocity=velocity,
                    vort_threshold=20,
                    q_threshold=5,
                    spacing=(2*np.pi/192, 1/65, np.pi/192),
                    cmap="jet",
                    save_path=save_path,
                )

            img = plt.imread(save_path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.figtext(0.5, 0.01, f"Q Isosurface (Q = 60)", ha='center', fontsize=16)
    final_path = f"generated_plots/{filename}.png"
    plt.savefig(final_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in {final_path}")
    
    method_qs = {
        "Reference": np.concatenate(qs_ref),
        "Supervised baseline": np.concatenate(qs_base),
        "Shu et al. method": np.concatenate(qs_shu),
        "Masked diffusion (ours)": np.concatenate(qs_diff),
    }

    plt.figure(figsize=(8, 5))

    # Define x-range for evaluation
    x_values = np.linspace(-50, 50, 1000)
    
    linestyles = ["solid", "dotted", "dashdot", "dashed"]

    for (label, q_values), ls in zip(method_qs.items(), linestyles):
        kde = gaussian_kde(q_values)
        density = kde(x_values)
        plt.plot(x_values, density, label=label, linewidth=2, linestyle=ls)

    plt.xlabel("Q", fontsize=16)
    plt.ylabel("p(Q)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title("Q-criterion Distribution per Method")
    plt.legend(fontsize=12)
    plt.grid(True)

    dist_path = f"generated_plots/q_distribution_kde_{filename}.png"
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Q-distribution KDE plot saved in {dist_path}")
    
    
def plot_report_8(dataset, config, indices, samples_x, samples_y, reg, ddpm, fm, dp, cond, hybrid, filename):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 3 * n))

    for row, idx in enumerate(indices):
        
        # Low-res
        axes[row, 0].imshow(
            samples_x[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 0].axis('off')
        
        # Regression
        axes[row, 1].imshow(
            reg[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 1].axis('off')
        
        # DDPM
        axes[row, 2].imshow(
            ddpm[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 2].axis('off')
        
        # FM
        axes[row, 3].imshow(
            fm[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 3].axis('off')
        
        # DP
        axes[row, 4].imshow(
            dp[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 4].axis('off')
        
        # Cond
        axes[row, 5].imshow(
            cond[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 5].axis('off')
        
        # Cond
        axes[row, 6].imshow(
            hybrid[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 6].axis('off')
        
        # Ground truth
        im = axes[row, 7].imshow(
            samples_y[idx, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 7].axis('off')
        
        # Leave the fourth subplot blank but place the colorbar there
        #axes[row, 7].axis('off')
        #cbar = fig.colorbar(im, ax=axes[row, 7], fraction=1.0)
        #cbar.ax.tick_params(labelsize=10)

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation", fontsize=16)
    axes[0, 1].set_title("P3D baseline", fontsize=16)
    axes[0, 2].set_title("DDPM mask method", fontsize=16)
    axes[0, 3].set_title("FM interp method", fontsize=16)
    axes[0, 4].set_title("FM DP method", fontsize=16)
    axes[0, 5].set_title("FM cond method", fontsize=16)
    axes[0, 6].set_title("Hybrid method", fontsize=16)
    axes[0, 7].set_title("Reference", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    
    compare_spectrum_7(reg, ddpm, fm, dp, cond, hybrid, samples_y, filename)
    
def plot_report_8_3d(dataset, config, indices, samples_x, samples_y, reg, ddpm, fm, dp, cond, hybrid, filename, norm=False, fluctuation=False):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 3 * n))
    
    if norm:
        samples_x = dataset.data_scaler(samples_x)
        samples_y = dataset.data_scaler(samples_y)
        reg = dataset.data_scaler(reg)
        ddpm = dataset.data_scaler(ddpm)
        fm = dataset.data_scaler(fm)
        dp = dataset.data_scaler(dp)
        hybrid = dataset.data_scaler(hybrid)
        cond = dataset.data_scaler(cond)
        
    if fluctuation:
        samples_x = samples_x - dataset.mean_velocity_field
        samples_y = samples_y - dataset.mean_velocity_field
        reg = reg - dataset.mean_velocity_field
        ddpm = ddpm - dataset.mean_velocity_field
        fm = fm - dataset.mean_velocity_field
        dp = dp - dataset.mean_velocity_field
        cond = cond - dataset.mean_velocity_field
        hybrid = hybrid - dataset.mean_velocity_field
        
    
    base_cmap = plt.get_cmap("turbo")
    listed_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    colormap = ramp_step_alpha(listed_cmap, 0.2, 0.99)

    for row, idx in enumerate(indices):
        # Low-res
        velocity = samples_x[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img0 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 0].imshow(img0)
        axes[row, 0].axis("off")
        
        # Regression
        velocity = reg[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img1 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 1].imshow(img1)
        axes[row, 1].axis("off")
        
        # DDPM Mask
        velocity = ddpm[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img2 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 2].imshow(img2)
        axes[row, 2].axis("off")
        
        # FM interp
        velocity = fm[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img3 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 3].imshow(img3)
        axes[row, 3].axis("off")
        
        # FM dp
        velocity = dp[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img4 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 4].imshow(img4)
        axes[row, 4].axis("off")
        
        # FM cond
        velocity = cond[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img5 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 5].imshow(img5)
        axes[row, 5].axis("off")
        
        # Hybrid
        velocity = hybrid[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img6 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 6].imshow(img6)
        axes[row, 6].axis("off")
        
        # Ground truth
        velocity = samples_y[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img7 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 7].imshow(img7)
        axes[row, 7].axis("off")
        
        np.save("data/q.npy", q)
        print(q.min(), q.max())

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation")
    axes[0, 1].set_title("P3D baseline")
    axes[0, 2].set_title("DDPM mask method")
    axes[0, 3].set_title("FM interp method")
    axes[0, 4].set_title("FM DP method")
    axes[0, 5].set_title("FM cond method")
    axes[0, 6].set_title("Hybrid method")
    axes[0, 7].set_title("Reference")

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    

def plot_report_8_3d_iso(dataset, config, indices, samples_x, samples_y, reg, ddpm, fm, dp, cond, hybrid, filename, norm=False, fluctuation=False, q_threshold=150):
    os.makedirs("generated_plots/frames", exist_ok=True)
    
    qs_interp = []
    qs_base = []
    qs_ddpm = []
    qs_fm = []
    qs_dp = []
    qs_cond = []
    qs_hybrid = []
    qs_ref = []
    
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 3 * n))
    
    if norm:
        samples_x = dataset.data_scaler(samples_x)
        samples_y = dataset.data_scaler(samples_y)
        reg = dataset.data_scaler(reg)
        ddpm = dataset.data_scaler(ddpm)
        fm = dataset.data_scaler(fm)
        dp = dataset.data_scaler(dp)
        hybrid = dataset.data_scaler(hybrid)
        cond = dataset.data_scaler(cond)
        
    if fluctuation:
        samples_x = samples_x - dataset.mean_velocity_field
        samples_y = samples_y - dataset.mean_velocity_field
        reg = reg - dataset.mean_velocity_field
        ddpm = ddpm - dataset.mean_velocity_field
        fm = fm - dataset.mean_velocity_field
        dp = dp - dataset.mean_velocity_field
        cond = cond - dataset.mean_velocity_field
        hybrid = hybrid - dataset.mean_velocity_field

    for row, idx in enumerate(indices):
        entries = [
            ("Interpolation", samples_x[idx]),
            ("P3D baseline", reg[idx]),
            ("DDPM mask method", ddpm[idx]),
            ("FM interp method", fm[idx]),
            ("FM DP method", dp[idx]),
            ("FM cond method", cond[idx]),
            ("Hybrid method", hybrid[idx]),
            ("Reference", samples_y[idx]),
        ]
        
        for col, (title, velocity_tensor) in enumerate(entries):
            velocity = velocity_tensor.unsqueeze(0)
            if dataset is not None:
                q = q_criterion(velocity, 128)
            else:
                q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
            
            q_flat = q[0, 0].detach().cpu().numpy().flatten()
            if col == 0:
                qs_interp.append(q_flat)
            if col == 1:
                qs_base.append(q_flat)
            if col == 2:
                qs_ddpm.append(q_flat)
            if col == 3:
                qs_fm.append(q_flat)
            if col == 4:
                qs_dp.append(q_flat)
            if col == 5:
                qs_cond.append(q_flat)
            if col == 6:
                qs_hybrid.append(q_flat)
            if col == 7:
                qs_ref.append(q_flat)
            
            save_path = f"generated_plots/frames/frame_{filename}_row{row}_col{col}.png"
            
            if dataset is not None:
                plot_q_isosurface(
                    Q_tensor=q,
                    velocity=velocity,
                    q_threshold=120,
                    vort_threshold=35,
                    spacing=(2*np.pi/128, 2*np.pi/128, 2*np.pi/128),
                    cmap="jet",
                    save_path=save_path,
                )
            else:
                plot_q_isosurface_ch(
                    Q_tensor=q,
                    velocity=velocity,
                    vort_threshold=20,
                    q_threshold=5,
                    spacing=(2*np.pi/192, 1/65, np.pi/192),
                    cmap="jet",
                    save_path=save_path,
                )
            
            img = plt.imread(save_path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            
            if row == 0:
                axes[row, col].set_title(title, fontsize=16)
                
        np.save("data/q.npy", q.cpu().numpy())
        print(f"Row {row}: Q min {q.min().item()}, max {q.max().item()}")

    plt.tight_layout()
    final_path = f"generated_plots/{filename}.png"
    plt.savefig(final_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in {final_path}")
    
    
    method_qs = {
        "Interpolation": np.concatenate(qs_interp),
        "P3D baseline": np.concatenate(qs_base),
        "Mask Diffusion": np.concatenate(qs_ddpm),
        "FM interp": np.concatenate(qs_fm),
        "Direct path": np.concatenate(qs_dp),
        "Conditioning": np.concatenate(qs_cond),
        "Hybrid": np.concatenate(qs_hybrid),
        "Reference": np.concatenate(qs_ref),
    }

    plt.figure(figsize=(10, 6))

    # Define x-range for evaluation
    x_values = np.linspace(-100, 100, 1000)

    for label, q_values in method_qs.items():
        q_pos = q_values
        kde = gaussian_kde(q_pos)
        density = kde(x_values)
        plt.plot(x_values, density, label=label, linewidth=2)

    plt.xlabel("Q value", fontsize=26)
    plt.ylabel("Probability Density", fontsize=26)
    plt.title("Q-criterion Distribution per Method", fontsize=26)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    dist_path = f"generated_plots/q_distribution_kde_{filename}.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"Q-distribution KDE plot saved in {dist_path}")

    
def compute_blurriness(tensor):
    """
    Compute variance of 3D Laplacian for each channel.
    tensor: numpy.ndarray of shape (C, Lx, Ly, Lz)
    Returns: numpy.ndarray of shape (C,) with Laplacian variances per channel
    """
    C = tensor.shape[0]
    variances = np.zeros(C)
    for c in range(C):
        lap = laplace(tensor[c])
        variances[c] = lap.var()
    return np.mean(variances)

def compute_vorticity(velocity, h, grid=None):
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"
    
    N = int(2 * torch.pi / h)
    if grid is None:
        mesh_grid = MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi, N), (0, 2*torch.pi, N)], device="cpu")
        velocity = velocity.to(mesh_grid.device)
    else:
        mesh_grid = grid
        
    rot = curl(velocity, mesh=mesh_grid)
    
    return rot

def compute_divergence(velocity, h, grid=None, div=None):
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"
    
    N = int(2 * torch.pi / h)
    if grid is None:
        mesh_grid = MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi, N), (0, 2*torch.pi, N)], device="cpu")
        velocity = velocity.to(mesh_grid.device)
    else:
        mesh_grid = grid
    
    #mesh_grid=MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi * 5 / 1024, 5), (0, 2*torch.pi, N)])
    
    if div is None:
        div_op=Div()
    else:
        div_op = div
        
    divergence = div_op(u=velocity, mesh=mesh_grid)
    
    return divergence

def compute_divergence_fdm(velocity, hx, hy, hz):
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"
    
    # Pad for central differences (replicate boundary)
    def central_diff(f, dim, delta):
        # f: (batch, Nx, Ny, Nz)
        pad = [0, 0, 0, 0, 0, 0]  # [z0, z1, y0, y1, x0, x1]
        pad[2 * (2 - dim) + 1] = 1  # after
        pad[2 * (2 - dim)] = 1      # before
        f_pad = torch.nn.functional.pad(f, pad, mode='replicate')
        # Central difference
        slices_before = [slice(None)] * 4
        slices_after = [slice(None)] * 4
        slices_before[dim+1] = slice(0, -2)
        slices_after[dim+1] = slice(2, None)
        return (f_pad[tuple(slices_after)] - f_pad[tuple(slices_before)]) / (2 * delta)

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    dvx_dx = central_diff(vx, 0, hx)
    dvy_dy = central_diff(vy, 1, hy)
    dvz_dz = central_diff(vz, 2, hz)

    divergence = dvx_dx + dvy_dy + dvz_dz
    
    return divergence.unsqueeze(0)

def compute_divergence_fdm_wall(velocity, hx, hy, hz):
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"
    
    def central_diff(f, dim, h):
        pad = [0, 0, 0, 0, 0, 0]  # [z0, z1, y0, y1, x0, x1]
        pad[2 * (2 - dim) + 1] = 1  # after
        pad[2 * (2 - dim)] = 1      # before
        f_pad = torch.nn.functional.pad(f, pad, mode='replicate')

        slices_before = [slice(None)] * 4
        slices_after = [slice(None)] * 4
        slices_before[dim + 1] = slice(0, -2)
        slices_after[dim + 1] = slice(2, None)
        return (f_pad[tuple(slices_after)] - f_pad[tuple(slices_before)]) / (2 * h)

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    dvx_dx = central_diff(vx, 0, hx)

    # Manual handling for vy (wall at y = y_max)
    batch_size, Nx, Ny, Nz = vy.shape
    dvy_dy = torch.zeros_like(vy)

    # Central differences for interior y
    dvy_dy[:, :, 1:-1, :] = (vy[:, :, 2:, :] - vy[:, :, :-2, :]) / (2 * hy)

    # Forward difference at y = 0 (open)
    dvy_dy[:, :, 0, :] = (vy[:, :, 1, :] - vy[:, :, 0, :]) / hy

    # Backward difference at y = Ny - 1 (wall: vy = 0)
    dvy_dy[:, :, -1, :] = (0.0 - vy[:, :, -2, :]) / hy

    dvz_dz = central_diff(vz, 2, hz)

    divergence = dvx_dx + dvy_dy + dvz_dz

    return divergence.unsqueeze(0)

def fourth_order_diff(f, dim, h):
    pad = [0, 0, 0, 0, 0, 0]  # [z0, z1, y0, y1, x0, x1]
    pad[2 * (2 - dim)] = 2     # before
    pad[2 * (2 - dim) + 1] = 2 # after
    #pad_dim = 2 * (2 - dim)
    #pad[pad_dim] = 2
    #pad[pad_dim + 1] = 2
    f_pad = torch.nn.functional.pad(f, pad, mode='replicate')

    slices = [slice(None)] * 4
    slices_m2 = slices.copy()
    slices_m1 = slices.copy()
    slices_p1 = slices.copy()
    slices_p2 = slices.copy()

    slices_m2[dim + 1] = slice(0, -4)
    slices_m1[dim + 1] = slice(1, -3)
    slices_p1[dim + 1] = slice(3, -1)
    slices_p2[dim + 1] = slice(4, None)

    return (-f_pad[tuple(slices_p2)] + 8*f_pad[tuple(slices_p1)] - 8*f_pad[tuple(slices_m1)] + f_pad[tuple(slices_m2)]) / (12 * h)


def yvector(ny):
    yF=np.cos(np.pi*np.arange(0,1+1/((ny-1)),1/((ny-1))))
    return 1 - yF

def compute_divergence_fdm_wall_high_order(velocity, hx, Ny, hz):
    """
    hy_vector: 1D tensor of shape (Ny-1,) with variable spacing between y-grid points
    """
    
    Nx = 192
    Nz = 192
    y = yvector(Ny)
    y = y/2
    dy = np.diff(y)
    dy = dy[1:]
    dy_tensor = torch.from_numpy(dy).float()
    
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    dvx_dx = fourth_order_diff(vx, 0, hx)
    dvz_dz = fourth_order_diff(vz, 2, hz)

    B, Nx, Ny, Nz = vy.shape
    dvy_dy = torch.zeros_like(vy)

    # Interior points: use fourth-order central differences
    for j in range(2, Ny - 2):
        h = (dy_tensor[j - 2] + dy_tensor[j - 1] + dy_tensor[j] + dy_tensor[j + 1]) / 4  # average spacing
        #h = (y[j+2] - y[j-2]) / 4
        dvy_dy[:, :, j, :] = (
            -vy[:, :, j + 2, :] + 8 * vy[:, :, j + 1, :] - 8 * vy[:, :, j - 1, :] + vy[:, :, j - 2, :]
        ) / (12 * h)

    # Forward difference at y = 0
    h0 = (dy_tensor[0] + dy_tensor[1]) / 2
    dvy_dy[:, :, 0, :] = (
        -3 * vy[:, :, 0, :] + 4 * vy[:, :, 1, :] - vy[:, :, 2, :]
    ) / (2 * h0)

    # Forward difference at y = 1
    h1 = (dy_tensor[1] + dy_tensor[2]) / 2
    dvy_dy[:, :, 1, :] = (
        -3 * vy[:, :, 1, :] + 4 * vy[:, :, 2, :] - vy[:, :, 3, :]
    ) / (2 * h1)

    # Backward difference at y = Ny - 2
    h_back2 = (dy_tensor[-3] + dy_tensor[-2]) / 2
    #h_back2 = y[-2] - y[-3]
    dvy_dy[:, :, -2, :] = (
        3 * vy[:, :, -2, :] - 4 * vy[:, :, -3, :] + vy[:, :, -4, :]
    ) / (2 * h_back2)

    # Backward difference at y = Ny - 1
    h_back1 = (dy_tensor[-2] + dy_tensor[-1]) / 2
    #h_back1 = y[-1] - y[-2]
    dvy_dy[:, :, -1, :] = (
        3 * vy[:, :, -1, :] - 4 * vy[:, :, -2, :] + vy[:, :, -3, :]
    ) / (2 * h_back1)

    divergence = dvx_dx + dvy_dy + dvz_dz
    return divergence.unsqueeze(0)


def compute_curl_fdm_wall_high_order(velocity, hx, Ny, hz):
    """
    Compute the curl of a 3D velocity field with 4th order FDM.
    Wall at y_max, y is non-uniform.
    Returns: curl with shape (B, 3, Nx, Ny, Nz)
    """
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    Nx = vx.shape[1]
    Nz = vx.shape[3]

    y = yvector(Ny)
    y = y / 2
    dy = np.diff(y)
    dy = dy[1:]  # remove first point since vy is defined on interior
    dy_tensor = torch.from_numpy(dy).float()

    B = vx.shape[0]
    curl = torch.zeros_like(velocity)

    # ∂vz/∂y and ∂vy/∂z
    dvz_dy = torch.zeros_like(vz)
    dvy_dz = fourth_order_diff(vy, dim=2, h=hz)

    for j in range(2, Ny - 3):
        h = (dy_tensor[j - 2] + dy_tensor[j - 1] + dy_tensor[j] + dy_tensor[j + 1]) / 4
        dvz_dy[:, :, j, :] = (
            -vz[:, :, j + 2, :] + 8 * vz[:, :, j + 1, :] - 8 * vz[:, :, j - 1, :] + vz[:, :, j - 2, :]
        ) / (12 * h)

    # Forward/backward at boundaries for dvz_dy
    dvz_dy[:, :, 0, :] = (-3 * vz[:, :, 0, :] + 4 * vz[:, :, 1, :] - vz[:, :, 2, :]) / (2 * (dy_tensor[0] + dy_tensor[1]) / 2)
    dvz_dy[:, :, 1, :] = (-3 * vz[:, :, 1, :] + 4 * vz[:, :, 2, :] - vz[:, :, 3, :]) / (2 * (dy_tensor[1] + dy_tensor[2]) / 2)
    dvz_dy[:, :, -2, :] = (3 * vz[:, :, -2, :] - 4 * vz[:, :, -3, :] + vz[:, :, -4, :]) / (2 * (dy_tensor[-3] + dy_tensor[-2]) / 2)
    dvz_dy[:, :, -1, :] = (3 * vz[:, :, -1, :] - 4 * vz[:, :, -2, :] + vz[:, :, -3, :]) / (2 * (dy_tensor[-2] + dy_tensor[-1]) / 2)

    curl[:, 0] = dvz_dy - dvy_dz

    # ∂vx/∂z and ∂vz/∂x
    dvx_dz = fourth_order_diff(vx, dim=2, h=hz)
    dvz_dx = fourth_order_diff(vz, dim=0, h=hx)
    curl[:, 1] = dvx_dz - dvz_dx

    # ∂vy/∂x and ∂vx/∂y
    dvy_dx = fourth_order_diff(vy, dim=0, h=hx)
    dvx_dy = torch.zeros_like(vx)

    for j in range(2, Ny - 3):
        h = (dy_tensor[j - 2] + dy_tensor[j - 1] + dy_tensor[j] + dy_tensor[j + 1]) / 4
        dvx_dy[:, :, j, :] = (
            -vx[:, :, j + 2, :] + 8 * vx[:, :, j + 1, :] - 8 * vx[:, :, j - 1, :] + vx[:, :, j - 2, :]
        ) / (12 * h)

    dvx_dy[:, :, 0, :] = (-3 * vx[:, :, 0, :] + 4 * vx[:, :, 1, :] - vx[:, :, 2, :]) / (2 * (dy_tensor[0] + dy_tensor[1]) / 2)
    dvx_dy[:, :, 1, :] = (-3 * vx[:, :, 1, :] + 4 * vx[:, :, 2, :] - vx[:, :, 3, :]) / (2 * (dy_tensor[1] + dy_tensor[2]) / 2)
    dvx_dy[:, :, -2, :] = (3 * vx[:, :, -2, :] - 4 * vx[:, :, -3, :] + vx[:, :, -4, :]) / (2 * (dy_tensor[-3] + dy_tensor[-2]) / 2)
    dvx_dy[:, :, -1, :] = (3 * vx[:, :, -1, :] - 4 * vx[:, :, -2, :] + vx[:, :, -3, :]) / (2 * (dy_tensor[-2] + dy_tensor[-1]) / 2)

    curl[:, 2] = dvy_dx - dvx_dy

    return curl  # shape (B, 3, Nx, Ny, Nz)


def upsample(data_lr, factor=2):
    data_lr = data_lr.float()
    N_time, N_channels, D, H, W = data_lr.shape
    data_lr_reshaped = data_lr.view(N_time * N_channels, 1, D, H, W)
    data_lr_upsampled = torch.nn.functional.interpolate(data_lr_reshaped, size=(D*factor, H*factor, W*factor), mode='nearest')
    velocity_lr_to_hr = data_lr_upsampled.view(N_time, N_channels, D*factor, H*factor, W*factor)
    print(f"velocity_lr upsampled to: {velocity_lr_to_hr.shape}")
    
    return velocity_lr_to_hr

def interpolate_points(image, perc=0, ids=None, method="nearest"):
    # Support both 2D and 3D images
    if image.ndim == 2:
        Nx, Ny = image.shape
        if ids is None:
            sampled_ids = random.sample(range(Nx * Ny), int(Nx * Ny * perc))
        else:
            sampled_ids = ids
        vals = np.tile(image.reshape(Nx * Ny)[sampled_ids], 9)
        ids = [[(x // Ny), (x % Ny)] for x in sampled_ids] + \
              [[(x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
              [[(x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids] + \
              [[Nx + (x // Ny), (x % Ny)] for x in sampled_ids] + \
              [[Nx + (x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
              [[Nx + (x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids] + \
              [[2*Nx + (x // Ny), (x % Ny)] for x in sampled_ids] + \
              [[2*Nx + (x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
              [[2*Nx + (x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids]
        grid_x, grid_y = np.mgrid[0:Nx*3, 0:Ny*3]
        grid_z = griddata(ids, vals, (grid_x, grid_y), method=method, fill_value=0)
        return torch.tensor(grid_z[Nx:Nx*2, Ny:Ny*2])
    elif image.ndim == 3:
        Nx, Ny, Nz = image.shape
        if ids is None:
            sampled_ids = random.sample(range(Nx * Ny * Nz), int(Nx * Ny * Nz * perc))
        else:
            sampled_ids = ids
        # Get the (x, y, z) coordinates for sampled indices
        coords = np.array([(idx // (Ny*Nz), (idx % (Ny*Nz)) // Nz, idx % Nz) for idx in sampled_ids])
        vals = image.reshape(-1)[sampled_ids]
        # Interpolate onto a dense grid
        grid_x, grid_y, grid_z = np.mgrid[0:Nx, 0:Ny, 0:Nz]
        interp = griddata(coords, vals, (grid_x, grid_y, grid_z), method=method, fill_value=0)
        return torch.tensor(interp)
        
    else:
        raise ValueError("Input image must be 2D or 3D.")

def interpolate_dataset(dataset, perc, method="nearest"):
    X_vals = dataset.cpu().clone() if type(dataset) is torch.Tensor else dataset.copy()
    n_samples = dataset.shape[0]
    n_channels = dataset.shape[1]
    dims = dataset.shape[2:]
    n_points = int(np.prod(dims) * perc)
    sampled_ids = np.zeros((n_samples, n_points), dtype=np.int32)
    
    random.seed(1234)

    for i in range(n_samples):
        print(f"sample {i+1}/{n_samples}")
        sampled_ids[i] = np.array(random.sample(range(np.prod(dims)), n_points))
        for c in range(n_channels):
            X_vals[i, c] = interpolate_points(X_vals[i, c], perc=perc, ids=sampled_ids[i], method=method)
    return X_vals, sampled_ids


def downscale_data(high_res, scale_factor, order=0):
    channels = len(high_res.shape) == 5  # (N, C, Lx, Ly, Lz)

    if channels:
        N, C, Lx, Ly, Lz = high_res.shape
        high_res = high_res.reshape(N * C, Lx, Ly, Lz)
    else:
        N, Lx, Ly, Lz = high_res.shape
        
    samples_ids = np.zeros((N, int(Lx*Ly*Lz * (1/4)**3)), dtype=np.int32)

    _high_res = high_res.numpy() if isinstance(high_res, torch.Tensor) else high_res

    Lx_small = int(Lx / scale_factor)
    Ly_small = int(Ly / scale_factor)
    Lz_small = int(Lz / scale_factor)
    NN = _high_res.shape[0]

    X_small = np.zeros((NN, Lx_small, Ly_small, Lz_small), dtype=np.float32)
    X_upscaled = np.zeros((NN, Lx, Ly, Lz), dtype=np.float32)
    
    # Compute coarse cell centers in the original grid
    cx = np.arange(scale_factor // 2, Lx, scale_factor)
    cy = np.arange(scale_factor // 2, Ly, scale_factor)
    cz = np.arange(scale_factor // 2, Lz, scale_factor)
    coarse_indices = []
    for ix in cx:
        for iy in cy:
            for iz in cz:
                coarse_indices.append(ix * (Ly * Lz) + iy * Lz + iz)
    coarse_indices = np.array(coarse_indices, dtype=np.int32)
    
    for i in range(N):
        samples_ids[i] = coarse_indices

    for i in range(NN):
        print(f"sample {i}/{NN}")
        # Downscale
        X_small[i] = zoom(_high_res[i], zoom=(Lx_small / Lx, Ly_small / Ly, Lz_small / Lz), order=order)
        # Upscale
        X_upscaled[i] = zoom(X_small[i], zoom=(Lx / Lx_small, Ly / Ly_small, Lz / Lz_small), order=order)

    if channels:
        X_upscaled = X_upscaled.reshape(N, C, Lx, Ly, Lz)

    return torch.Tensor(X_upscaled), samples_ids


def LSiM_distance_3D(A, B):
    A = A.squeeze(0)
    A = A.permute(1, 2, 3, 0)
    A = A.cpu().numpy()
    B = B.squeeze(0)
    B = B.permute(1, 2, 3, 0)
    B = B.cpu().numpy()
    model_3d = DistanceModel.load("LSIM_3D/models/VolSiM.pth", useGPU=False)
    dist = model_3d.computeDistance(A, B, normalize=True, interpolate=False)
    
    return dist[0]

def diffuse_mask(value_ids, A=1, sig=0.044, search_dist=-1, N=256, Nx=256, Ny=256, Nz=None, tol=1e-6, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi):
    """
    Create a 2D or 3D diffuse mask with Gaussian spread around value_ids.
    If Nz is None, defaults to 2D (Nx, Ny). If Nz is given, mask is 3D (Nx, Ny, Nz).
    """
    dx = Lx / Nx
    dy = Ly / Ny
    if Nz is not None:
        dz = Lz / Nz
        grid = np.zeros((Nx, Ny, Nz))
        # Set boundaries to 1
        grid[0, :, :] = 1
        grid[-1, :, :] = 1
        grid[:, 0, :] = 1
        grid[:, -1, :] = 1
        grid[:, :, 0] = 1
        grid[:, :, -1] = 1

        def gauss3d(x0, y0, z0, x, y, z):
            return A * np.exp(-((x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2) / (2 * sig**2))

        if search_dist < 0:
            min_search_steps = 0
            while gauss3d(0, 0, 0, dx*min_search_steps, 0, 0) > tol:
                min_search_steps += 1
            search_dist = min_search_steps

        S = search_dist * 2 + 1
        gaussian = np.zeros((S, S, S))
        x0 = y0 = z0 = search_dist * dx
        for i in range(S):
            for j in range(S):
                for k in range(S):
                    gaussian[i, j, k] = gauss3d(x0, y0, z0, i*dx, j*dy, k*dz)

        for sid in value_ids:
            i = sid // (Ny * Nz)
            j = (sid % (Ny * Nz)) // Nz
            k = sid % Nz

            ilb = max(0, i - search_dist)
            iub = min(Nx, i + search_dist + 1)
            jlb = max(0, j - search_dist)
            jub = min(Ny, j + search_dist + 1)
            klb = max(0, k - search_dist)
            kub = min(Nz, k + search_dist + 1)

            gilb = max(0, search_dist - i)
            giub = S - max(0, i + search_dist - (Nx - 1))
            gjlb = max(0, search_dist - j)
            gjub = S - max(0, j + search_dist - (Ny - 1))
            gklb = max(0, search_dist - k)
            gkub = S - max(0, k + search_dist - (Nz - 1))

            grid[ilb:iub, jlb:jub, klb:kub] = np.fmax(
                gaussian[gilb:giub, gjlb:gjub, gklb:gkub],
                grid[ilb:iub, jlb:jub, klb:kub]
            )
        return grid
    else:
        print("No 3D data")
    
# https://github.com/atong01/conditional-flow-matching/blob/c25e1918a80dfacbe9475c055d61ac997f28262a/torchcfm/optimal_transport.py#L218
def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret

def init_weights(model):
    """
    Set weight initialization for Conv3D in network.
    Based on: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24
    """
    if isinstance(model, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.constant_(model.bias, 0)
        # torch.nn.init.zeros_(model.bias)
        
def spectral_derivative_3d(V):
    N, C, H, W, D = V.size()

    # Generate frequency grids
    h = np.fft.fftfreq(H, 1. / H)
    w = np.fft.fftfreq(W, 1. / W)
    d = np.fft.fftfreq(D, 1. / D)
    mesh_h, mesh_w, mesh_d = np.meshgrid(h, w, d, indexing='ij')

    # Convert to torch tensors
    mesh_h = torch.tensor(mesh_h, device=V.device, dtype=V.dtype)
    mesh_w = torch.tensor(mesh_w, device=V.device, dtype=V.dtype)
    mesh_d = torch.tensor(mesh_d, device=V.device, dtype=V.dtype)

    # Perform FFT
    V_fft = torch.fft.fftn(V, dim=(-3, -2, -1))

    # Multiply by i * k to compute derivative in Fourier domain
    I = 1j
    dV_dh_fft = I * mesh_h * V_fft
    dV_dw_fft = I * mesh_w * V_fft
    dV_dd_fft = I * mesh_d * V_fft

    # Inverse FFT to get real-valued derivatives
    dV_dh = torch.fft.ifftn(dV_dh_fft, dim=(-3, -2, -1)).real
    dV_dw = torch.fft.ifftn(dV_dw_fft, dim=(-3, -2, -1)).real
    dV_dd = torch.fft.ifftn(dV_dd_fft, dim=(-3, -2, -1)).real

    # Stack derivatives along new dimension
    dV = torch.stack([dV_dh, dV_dw, dV_dd], dim=2)

    return dV


def physics(A_model, A_target):
    # continuity = [None, None]
    S_ijS_ij_m = [None, None]
    R_ijR_ij_m = [None, None]
    SijSkjSji_m = [None, None]
    VortexStret_m = [None, None]
    A_model = A_model[:, :3, :, :, :]
    A_target = A_target[:, :3, :, :, :]
    for i, A in enumerate([A_model, A_target]):
        A11, A22, A33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
        # continuity[i] = (A11 + A22 + A33).mean()
        S = 0.5 * (A + A.transpose(1, 2))
        R = 0.5 * (A - A.transpose(1, 2))
        S_ijS_ij = (S * S).sum(dim=[1, 2])
        R_ijR_ij = (R * R).sum(dim=[1, 2])
        S_ijS_ij_m[i] = S_ijS_ij.mean()
        R_ijR_ij_m[i] = R_ijR_ij.mean()

        S = S.permute(0, 3, 4, 5, 1, 2).reshape(-1, 3, 3)
        R = R.permute(0, 3, 4, 5, 1, 2).reshape(-1, 3, 3)
        SijSkjSji = torch.sum(torch.matmul(S, S) * S, axis=(1, 2))
        Omega = torch.empty((*R.size()[:-1], 1), device=R.device)
        Omega[:, 0, 0] = 2 * R[:, 2, 1]
        Omega[:, 1, 0] = 2 * R[:, 0, 2]
        Omega[:, 2, 0] = 2 * R[:, 1, 0]
        VS_3d = torch.matmul(S, Omega)
        VortexStret = torch.matmul(Omega.transpose(1, 2), VS_3d)
        SijSkjSji_m[i] = SijSkjSji.mean()
        VortexStret_m[i] = (-3 / 4) * VortexStret.mean()

    weight = torch.tensor([1, 1, 1, 1], device=A_model.device)
    output = 0
    for i, item in enumerate([S_ijS_ij_m, R_ijR_ij_m, SijSkjSji_m, VortexStret_m]):
        output += (item[1] - item[0]).abs() * weight[i]

    return output


def weighted_mse_loss(input, target, weight=(2. * torch.ones(3, 3)).fill_diagonal_(1)):
    loss_ = nn.functional.mse_loss(input, target, reduction='none')
    return sum([(weight[i, j] * loss_[:, i, j]).mean() for i in range(3) for j in range(3)])

def visualize_3d_cloud_volume(
    volume_data: torch.Tensor | np.ndarray,
    title: str = "3D Cloud Volume Rendering",
    bg_color: str = 'black',
    scalars_min: float = None,
    scalars_max: float = None
):
    """
    Visualizes a 3D scalar volumetric field as a cloud-like volume rendering
    using vedo. The color and density of the cloud indicate the data values.

    Args:
        volume_data (torch.Tensor or np.ndarray):
            The 3D scalar field data. Expected shape (D, H, W) or (1, D, H, W).
            If a PyTorch tensor, it will be moved to CPU and converted to NumPy.
        title (str): Title for the visualization window.
        bg_color (str): Background color of the plot ('black', 'white', etc.).
        scalars_min (float, optional): Optional minimum value for mapping data
            to the transfer function. If None, uses data_np.min().
        scalars_max (float, optional): Optional maximum value for mapping data
            to the transfer function. If None, uses data_np.max().
    """
    # Ensure vedo is installed
    try:
        import vedo
    except ImportError:
        print("Error: vedo library not found. Please install it using: pip install vedo")
        return

    # Convert input to a 3D NumPy array (D, H, W)
    if isinstance(volume_data, torch.Tensor):
        # Remove batch dimension if present and move to CPU, then convert to NumPy
        if volume_data.ndim == 4 and volume_data.shape[0] == 1:
            data_np = volume_data.squeeze(0).detach().cpu().numpy()
        elif volume_data.ndim == 3:
            data_np = volume_data.detach().cpu().numpy()
        else:
            raise ValueError(f"Input tensor has unsupported shape: {volume_data.shape}. Expected (D, H, W) or (1, D, H, W).")
    elif isinstance(volume_data, np.ndarray):
        if volume_data.ndim == 4 and volume_data.shape[0] == 1:
            data_np = volume_data.squeeze(0)
        elif volume_data.ndim == 3:
            data_np = volume_data
        else:
            raise ValueError(f"Input numpy array has unsupported shape: {volume_data.shape}. Expected (D, H, W) or (1, D, H, W).")
    else:
        raise TypeError("Input data must be a torch.Tensor or numpy.ndarray.")

    # Determine data range for normalization
    data_min = scalars_min if scalars_min is not None else data_np.min()
    data_max = scalars_max if scalars_max is not None else data_np.max()

    if data_max == data_min:
        print("Warning: All values in the volume data are identical. Cannot create meaningful visualization.")
        return

    # Normalize data to [0, 1] for consistent transfer function mapping
    # This step is crucial if your data's actual range can vary significantly.
    normalized_data_np = (data_np - data_min) / (data_max - data_min)

    # Create a vedo Volume object
    vol = vedo.Volume(normalized_data_np)

    # Define the custom transfer function for "cloud-like" rendering
    # These scalar values correspond to the normalized data range [0, 1]
    scalars = [0.0, 0.05, 0.15, 0.4, 0.7, 1.0]

    # Opacities (0.0 = fully transparent, 1.0 = fully opaque)
    # Creates the cloud effect: fuzzy at edges, denser in core
    opacities = [0.0, 0.005, 0.05, 0.2, 0.5, 0.8]

    # Colors (define the colormap from cool to hot)
    colors = [
        'lightskyblue', # Lowest values (most transparent blue)
        'cyan',
        'lime',
        'yellow',
        'orange',
        'red'           # Highest values (most opaque red)
    ]

    # Build the Lookup Table (LUT)
    transfer_function = vedo.build_lut(scalars, opacities, colors)

    # Apply the transfer function to the volume
    vol.cmap(transfer_function)
    vol.mode('composite') # Ensure composite rendering

    # Set up the plotter and visualize
    plotter = vedo.Plotter(size=(900, 900), bg=bg_color)

    # Add scalar bar for reference
    # Use the original data range for the scalar bar labels
    vol.add_scalarbar(f"Field Value ({data_min:.2f} to {data_max:.2f})", c='white' if bg_color == 'black' else 'black')

    # Add lighting for better depth perception
    vol.lighting('plastic')

    # Add the volume to the plotter
    plotter.add(vol)

    # Set an initial camera position. You can adjust these values
    # The default view often works well, but this provides a consistent starting point.
    center = np.array(data_np.shape) / 2
    plotter.camera.SetPosition([center[0]*2.5, center[1]*2.5, center[2]*2.5]) # View from outside
    plotter.camera.SetFocalPoint(center) # Look at the center of the volume
    plotter.camera.SetViewUp([0, 1, 0]) # Keep Y-axis up

    # Show the plot
    plotter.show(
        title,
        interactor_style=1, # 1 for TrackballCamera, allows easy navigation
    )

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def compute_energy_spectrum_2D(velocity, name, device, lx=2.0*math.pi, ly=2.0*math.pi, smooth=True, plot=False):
    velocity = velocity.to(device)
        
    N, _, nx, ny = velocity.shape
    nt = nx * ny

    k0x = 2.0 * math.pi / lx
    k0y = 2.0 * math.pi / ly
    knorm = (k0x + k0y) / 2.0
    n = max(nx, ny)
    wave_numbers = knorm * torch.arange(0, n, device=device)

    spectra = []

    for i in range(N):
        u = velocity[i, 0]
        v = velocity[i, 1]

        uh = torch.fft.fft2(u) / nt
        vh = torch.fft.fft2(v) / nt

        tkeh = 0.5 * (uh.abs()**2 + vh.abs()**2)

        kx = torch.fft.fftfreq(nx, d=1.0 / nx, device=device)
        ky = torch.fft.fftfreq(ny, d=1.0 / ny, device=device)
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        rk = torch.sqrt(kx**2 + ky**2)

        tke_spectrum = torch.zeros(len(wave_numbers), device=device)
        for k in range(len(wave_numbers)):
            mask = (rk >= wave_numbers[k]) & (rk < wave_numbers[k] + knorm)
            tke_spectrum[k] = tkeh[mask].sum()

        tke_spectrum /= knorm
        tke_spectrum = tke_spectrum[1:]

        if smooth:
            smoothed = torch.tensor(movingaverage(tke_spectrum.cpu(), 5), device=device)
            smoothed[:4] = tke_spectrum[:4]
            tke_spectrum = smoothed

        spectra.append(tke_spectrum)

    tke_spectrum_avg = torch.stack(spectra).mean(dim=0)
    wave_numbers = wave_numbers[1:]

    # Analytical line (2D turbulence: typically E(k) ~ k^(-3) in enstrophy cascade)
    C = 1.6
    eps = 0.103
    E_k_analytic = C * (eps ** (2/3)) * (wave_numbers ** (-5/3))

    if plot:
        plt.figure(figsize=(10, 4))
        plt.loglog(wave_numbers.cpu(), tke_spectrum_avg.cpu() + 1e-20, label="Averaged TKE Spectrum")
        plt.loglog(wave_numbers.cpu(), E_k_analytic.cpu(), 'k--', label=r"$1.6 \, \varepsilon^{2/3} \, k^{-5/3}$")
        plt.xlabel("Wavenumber $k$", fontsize=18)
        plt.ylabel("Energy $E(k)$", fontsize=18)
        plt.ylim(1e-7, 1)
        plt.title("2D Energy Spectrum", fontsize=18)
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        output_file = f"generated_plots/{name}_2D.png"
        plt.savefig(output_file, dpi=500)
        plt.close()

    return tke_spectrum_avg, wave_numbers

def compute_energy_spectrum(velocity, name, device, lx=2.0*math.pi, ly=2.0*math.pi, lz=2.0*math.pi, smooth=True, plot=False):
    
    velocity = velocity.to(device)
    N, _, nx, ny, nz = velocity.shape
    nt = nx * ny * nz

    # Grid spacing
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # Fundamental wavenumbers
    k0x = 2.0 * math.pi / lx
    k0y = 2.0 * math.pi / ly
    k0z = 2.0 * math.pi / lz

    # Frequencies in each direction
    kx = 2 * math.pi * torch.fft.fftfreq(nx, d=dx, device=device)
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=dy, device=device)
    kz = 2 * math.pi * torch.fft.fftfreq(nz, d=dz, device=device)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")
    rk = torch.sqrt(kx**2 + ky**2 + kz**2)

    # Bin setup
    dk = min(k0x, k0y, k0z)  # bin width = smallest fundamental mode
    kmax = rk.max().item()
    wave_numbers = torch.arange(0, kmax, dk, device=device)

    spectra = []

    for i in range(N):
        u = velocity[i, 0]
        v = velocity[i, 1]
        w = velocity[i, 2]

        # Perform FFT
        uh = torch.fft.fftn(u) / nt
        vh = torch.fft.fftn(v) / nt
        wh = torch.fft.fftn(w) / nt

        # Kinetic energy in Fourier space
        tkeh = 0.5 * (uh.abs()**2 + vh.abs()**2 + wh.abs()**2)

        # Bin energy
        tke_spectrum = torch.zeros(len(wave_numbers), device=device)
        for k in range(len(wave_numbers)):
            mask = (rk >= wave_numbers[k]) & (rk < wave_numbers[k] + dk)
            tke_spectrum[k] = tkeh[mask].sum()

        # Normalize by bin width
        tke_spectrum /= dk
        tke_spectrum = tke_spectrum[1:]  # skip k=0

        # Optional smoothing
        if smooth and len(tke_spectrum) > 5:
            smoothed = movingaverage(tke_spectrum.cpu(), 5)
            smoothed = torch.tensor(smoothed, device=device)
            smoothed[:4] = tke_spectrum[:4]  # keep low-k values
            tke_spectrum = smoothed

        spectra.append(tke_spectrum)

    # Average over snapshots
    tke_spectrum_avg = torch.stack(spectra).mean(dim=0)
    wave_numbers = wave_numbers[1:]  # match truncation above

    # Kolmogorov -5/3 line (for reference)
    C = 1.6
    eps = 0.103
    E_k_analytic = C * (eps ** (2/3)) * (wave_numbers ** (-5/3))

    # Plot
    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(wave_numbers.cpu(), tke_spectrum_avg.cpu() + 1e-20, 'r', label="$E(k)$")
        plt.loglog(wave_numbers.cpu(), E_k_analytic.cpu(), 'k--', 
                   label=r"$1.6 \, \varepsilon^{2/3} \, k^{-5/3}$")
        plt.xlabel("$k$", fontsize=22)
        plt.ylabel("$E(k)$", fontsize=22)
        plt.ylim(1e-7, 1)
        #plt.title("Energy Spectrum", fontsize=22)
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend(fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        output_file = f"generated_plots/{name}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

    return tke_spectrum_avg, wave_numbers

def compare_spectrums(E_gt, E, k, name):
    plt.figure(figsize=(10, 4))
    plt.loglog(k.cpu(), E_gt.cpu() + 1e-20, label="GT Spectrum")
    plt.loglog(k.cpu(), E.cpu() + 1e-20, label="Recons Spectrum")
    plt.xlabel("Wavenumber $k$", fontsize=18)
    plt.ylabel("Energy $E(k)$", fontsize=18)
    plt.ylim(1e-7, 1)
    plt.title("Energy Spectrum", fontsize=18)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    output_file = f"generated_plots/{name}.png"
    plt.savefig(output_file, dpi=500)
    plt.close()
    
def compare_spectrums_datasets(E_iso, k_iso, E_ch, k_ch, name):
    # Kolmogorov -5/3 line (for reference)
    C = 1.6
    eps = 0.103
    E_k_analytic = C * (eps ** (2/3)) * (k_iso ** (-5/3))
    
    plt.figure(figsize=(10, 4))
    plt.loglog(k_iso.cpu(), E_iso.cpu() + 1e-20, label="Isotropic Turbulence")
    plt.loglog(k_ch.cpu(), E_ch.cpu() + 1e-20, label="3D Channel")
    plt.loglog(k_iso.cpu(), E_k_analytic.cpu(), 'k--', 
                   label=r"$1.6 \, \varepsilon^{2/3} \, k^{-5/3}$")
    plt.xlabel("Wavenumber $k$", fontsize=18)
    plt.ylabel("Energy $E(k)$", fontsize=18)
    plt.ylim(1e-10, 1)
    plt.title("Energy Spectrum Comparison", fontsize=18)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    output_file = f"generated_plots/{name}.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    
def compare_spectrum_4(baseline, shu, mask, ref, name):
    e_base, k_base = compute_energy_spectrum(baseline, "baseline", baseline.device)
    e_shu, k_shu = compute_energy_spectrum(shu, "shu", shu.device)
    e_mask, k_mask = compute_energy_spectrum(mask, "mask", mask.device)
    e_ref, k_ref = compute_energy_spectrum(ref, "ref", ref.device)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(k_base.cpu(), e_base.cpu() + 1e-20, label="Supervised baseline")
    ax.loglog(k_shu.cpu(), e_shu.cpu() + 1e-20, label="Shu et al. method")   
    ax.loglog(k_mask.cpu(), e_mask.cpu() + 1e-20, label="Masked diffusion (ours)")
    ax.loglog(k_ref.cpu(), e_ref.cpu() + 1e-20, label="Reference") 
    
    ax.set_xlabel("Wavenumber $k$", fontsize=18)
    ax.set_ylabel("Energy $E(k)$", fontsize=18)
    ax.set_ylim(1e-8, 1)
    ax.set_title("Energy Spectrum", fontsize=18)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend(fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    
    # === Add inset (zoom box) ===
    axins = inset_axes(ax, width="30%", height="50%", loc='lower center')  
    axins.loglog(k_base.cpu(), e_base.cpu() + 1e-20)
    axins.loglog(k_shu.cpu(), e_shu.cpu() + 1e-20)   
    axins.loglog(k_mask.cpu(), e_mask.cpu() + 1e-20)
    axins.loglog(k_ref.cpu(), e_ref.cpu() + 1e-20)

    # Define zoom region (adjust to your data!)
    x1, x2 = 5, 7
    y1, y2 = 6e-3, 3e-2
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.xaxis.set_major_locator(NullLocator())
    axins.yaxis.set_major_locator(NullLocator())
    axins.xaxis.set_minor_locator(NullLocator())
    axins.yaxis.set_minor_locator(NullLocator())

    # Draw rectangle + connectors
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.tight_layout()
    output_file = f"generated_plots/{name}_freq.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    
    
def compare_spectrum_7(baseline, ddpm, fm, dp, cond, hybrid, ref, name):
    e_base, k_base = compute_energy_spectrum(baseline, "baseline", baseline.device)
    e_ddpm, k_ddpm = compute_energy_spectrum(ddpm, "ddpm", ddpm.device)
    e_fm, k_fm = compute_energy_spectrum(fm, "fm", fm.device)
    e_dp, k_dp = compute_energy_spectrum(dp, "dp", dp.device)
    e_cond, k_cond = compute_energy_spectrum(cond, "cond", cond.device)
    e_hybrid, k_hybrid = compute_energy_spectrum(hybrid, "hybrid", hybrid.device)
    e_ref, k_ref = compute_energy_spectrum(ref, "ref", ref.device)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(k_base.cpu(), e_base.cpu() + 1e-20, label="P3D baseline")
    plt.loglog(k_ddpm.cpu(), e_ddpm.cpu() + 1e-20, label="DDPM mask method")
    plt.loglog(k_fm.cpu(), e_fm.cpu() + 1e-20, label="FM interp method")
    plt.loglog(k_dp.cpu(), e_dp.cpu() + 1e-20, label="FM DP method")
    plt.loglog(k_cond.cpu(), e_cond.cpu() + 1e-20, label="FM Cond method")
    plt.loglog(k_hybrid.cpu(), e_hybrid.cpu() + 1e-20, label="Hybrid method")
    plt.loglog(k_ref.cpu(), e_ref.cpu() + 1e-20, label="Reference") 
    
    plt.xlabel("Wavenumber $k$", fontsize=26)
    plt.ylabel("Energy $E(k)$", fontsize=26)
    plt.ylim(1e-8, 1)
    plt.title("Energy Spectrum", fontsize=26)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    output_file = f"generated_plots/{name}_freq.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    
def compute_energy_spectrum_1d(velocity, name, device, index):
    N, C, nx, ny, nz = velocity.shape
    spectra = []
    for i in range(N):
        u_fm = velocity[i, :, :, index, :] #shape (3, 192, 192)
        nz_ = u_fm.shape[-1]
        kz_rfft = np.arange(1,nz_ / 2 + 0.1, 1)
        fft_fm = kz_rfft * np.mean(np.abs(np.fft.rfft(u_fm.cpu().numpy(), axis=-1))**2, axis=(0,1))[:int(nz_ / 2)]  # shape (192, 97)
        E_fm = np.max(fft_fm)
        Energy_fm = fft_fm / E_fm
        spectra.append(torch.tensor(Energy_fm, device=device))
        
    tke_spectrum_avg = torch.stack(spectra).mean(dim=0)
    
    return tke_spectrum_avg, torch.tensor(kz_rfft, device=device)

def compute_energy_spectra_2d(velocity_ch, device):
    xl = 4*np.pi
    zl = 2*np.pi
    nx_ = 192
    nz_ = 192
    ny_ = 65
    Re_cl = 2100
    nu = 1/Re_cl
    u_tau = 0.04285714285714286 #(Re_tau*nu)/2
    lstar=nu/u_tau
    N_samples = velocity_ch.shape[0]

    n_comp = 3
    y = yvector(65)
    y_pos = y #/ 2
    yp = y_pos / lstar
    print(yp)
    kz_rfft = np.arange(1,nz_ / 2 + 0.1, 1)
    spectra_u = np.ndarray((N_samples, n_comp, kz_rfft.shape[0], ny_ - 1),dtype="float")
    stds = [0.3810915451115368, 0.1435627775092393, 0.21601074762228595]
    velocity_ch = velocity_ch.cpu().numpy()
    for i in range(N_samples):
        for i_comp in range(n_comp):
            gen_u = stds[i_comp] * velocity_ch[:, i_comp, :, ::-1, :] 
            #gen_u = velocity_ch[:, i_comp, :, ::-1, :]  # shape (1, Nx, Ny, Nz)
            for i_y in range(ny_ - 1):
                # Compute spectrum for each wall-normal position
                spectrum = kz_rfft * np.mean(np.abs(np.fft.rfft(gen_u[i, :, i_y, :], axis=-1))**2, axis=0)[:int(nz_ / 2)]  # shape (kz_rfft,)
                spectra_u[i, i_comp, :, i_y] = spectrum
    
    spectra_data = np.mean(spectra_u, axis=0)  # shape (3, kz_rfft, ny_)
    lamdaz = 1 / (kz_rfft / (2*zl)) / lstar
    #spectra_u = np.nan_to_num(spectra_u, nan=0.0, posinf=0.0, neginf=0.0)
    Emax = np.max(spectra_data, axis=(1, 2))
    #Emax = np.where((Emax == 0) | np.isnan(Emax), 1.0, Emax)
    
    return lamdaz, yp, spectra_data, Emax
    
def compare_spectrum_7_ch(baseline, ddpm, fm, dp, cond, hybrid, ref, name):
    e_base, k_base = compute_energy_spectrum_1d(baseline, "baseline", baseline.device, 51)
    e_ddpm, k_ddpm = compute_energy_spectrum_1d(ddpm, "ddpm", ddpm.device, 51)
    e_fm, k_fm = compute_energy_spectrum_1d(fm, "fm", fm.device, 51)
    e_dp, k_dp = compute_energy_spectrum_1d(dp, "dp", dp.device, 51)
    e_cond, k_cond = compute_energy_spectrum_1d(cond, "cond", cond.device, 51)
    e_hybrid, k_hybrid = compute_energy_spectrum_1d(hybrid, "hybrid", hybrid.device, 51)
    e_ref, k_ref = compute_energy_spectrum_1d(ref, "ref", ref.device, 51)
    
    zl = 2*np.pi
    nx_ = 192
    nz_ = 192
    ny_ = 65
    Re_cl = 2100
    nu = 1/Re_cl
    u_tau = 0.04285714285714286 #(Re_tau*nu)/2
    lstar=nu/u_tau
    #k_base = 1 / (k_base / (2*zl)) / lstar
    #k_ddpm = 1 / (k_ddpm / (2*zl)) / lstar
    #k_fm = 1 / (k_fm / (2*zl)) / lstar
    #k_dp = 1 / (k_dp / (2*zl)) / lstar
    #k_cond = 1 / (k_cond / (2*zl)) / lstar
    #k_hybrid = 1 / (k_hybrid / (2*zl)) / lstar
    #k_ref = 1 / (k_ref / (2*zl)) / lstar

    plt.figure(figsize=(10, 6))
    plt.loglog(k_base[:-2].cpu(), e_base[:-2].cpu() + 1e-20, label="P3D baseline")
    plt.loglog(k_ddpm[:-2].cpu(), e_ddpm[:-2].cpu() + 1e-20, label="DDPM mask method")
    plt.loglog(k_fm[:-2].cpu(), e_fm[:-2].cpu() + 1e-20, label="FM interp method")
    plt.loglog(k_dp[:-2].cpu(), e_dp[:-2].cpu() + 1e-20, label="FM DP method")
    plt.loglog(k_cond[:-2].cpu(), e_cond[:-2].cpu() + 1e-20, label="FM Cond method")
    plt.loglog(k_hybrid[:-2].cpu(), e_hybrid[:-2].cpu() + 1e-20, label="Hybrid method")
    plt.loglog(k_ref[:-2].cpu(), e_ref[:-2].cpu() + 1e-20, label="Reference") 
    
    plt.xlabel("Wavenumber k", fontsize=26)
    plt.ylabel("Energy $E(k)$", fontsize=26)
    plt.ylim(1e-8, 1)
    plt.title("Energy Spectrum", fontsize=26)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    output_file = f"generated_plots/{name}_freq.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    
    
    # Compute 2D spectra for all methods (for contour plotting)
    methods = {
        "P3D baseline": baseline,
        "DDPM mask": ddpm,
        #"FM interp": fm,
        #"FM DP": dp,
        #"FM cond": cond,
        #"Hybrid": hybrid,
        "Reference": ref
    }

    spectra_all = {}
    for method_name, method_data in methods.items():
        lamdaz, yp, spectra_u, Emax = compute_energy_spectra_2d(method_data, method_data.device)
        spectra_all[method_name] = (spectra_u, Emax)

    # Now plot contours for all methods overlaid in each component subplot
    n_comp = 3
    fig, ax = plt.subplots(1, n_comp, sharey=True, figsize=[19, 6.4])
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # One color per level (5 levels)
    linestyles = ['dotted', 'dashed', 'solid']  # One linestyle per method
    method_names = list(methods.keys())

    for i_comp in range(n_comp):
        ax[i_comp].set(xscale="log", yscale="log", ylim=[max(1.0, yp.min()), yp.max()], xlabel=r'$\lambda^+_z$')
        ax[i_comp].set_title(f"Component {i_comp + 1}", fontsize=20)
        ax[i_comp].set_xlabel(r'$\lambda^+_z$', fontsize=20)
        ax[i_comp].tick_params(axis='both', labelsize=18) 
        
        # Determine global vmin/vmax across all methods for this component (for consistent scaling)
        all_contour_data = []
        Emax_global = spectra_all["Reference"][1]
        for method_name in method_names:
            spectra_u, Emax = spectra_all[method_name]
            contour_data = spectra_u[i_comp].T[:, 1:] / Emax_global[i_comp]
            all_contour_data.append(contour_data)
        global_min = min([np.nanpercentile(data, 5) for data in all_contour_data])
        global_max = max([np.nanpercentile(data, 95) for data in all_contour_data])
        if global_min == global_max:
            global_min, global_max = min([data.min() for data in all_contour_data]), max([data.max() for data in all_contour_data])
        
        # Plot contours for each method overlaid
        for idx, method_name in enumerate(method_names):
            spectra_u, Emax = spectra_all[method_name]
            contour_data = spectra_u[i_comp].T[:, :] / Emax[i_comp]
            contour_data = np.ma.masked_invalid(contour_data)
            
            #levels = np.linspace(global_min, global_max, 4)
            levels = [0.1, 0.5, 0.8]
            
            c = ax[i_comp].contour(lamdaz, yp[1:], contour_data, levels=levels, colors=colors, linestyles=linestyles[idx], linewidths=1.5, label=method_name)
        
        # Add custom legend to the first subplot only (shows method names with their linestyles)
        if i_comp == 1:
            handles = []
            for idx, method_name in enumerate(method_names):
                # Create a dummy line for each method's linestyle
                handles.append(plt.Line2D([0], [0], color='black', linestyle=linestyles[idx], linewidth=2, label=method_name))
            ax[i_comp].legend(handles=handles, loc='lower left', fontsize=18)

    ax[0].set_ylabel(r'$y^+$', fontsize=20)
    fig.tight_layout()
    path = f"generated_plots/{name}_2Dspectra.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    
    
    
def compute_energy_spectrum_original(file_path: str, name: str, smooth=True):
    lx = ly = lz = 2.0 * math.pi

    with h5py.File(file_path, 'r') as f:
        keys = list(f['sims']['sim0'].keys())[:5]  
        print(f"Found {len(keys)} snapshots.")
        
        spectra = []
        for i, key in enumerate(keys):
            print(f"\n📦 Processing snapshot {i+1}/{len(keys)}: {key}")
            sample = f['sims']['sim0'][key]
            velocity = np.transpose(sample, (3, 0, 1, 2))[:3]  # (3, Nx, Ny, Nz)

            # Add batch dimension: (1, 3, Nx, Ny, Nz)
            velocity = velocity[np.newaxis, ...]

            N, _, nx, ny, nz = velocity.shape
            nt = nx * ny * nz
            n = nx
            
            k0x = 2.0 * math.pi / lx
            k0y = 2.0 * math.pi / ly
            k0z = 2.0 * math.pi / lz
            knorm = (k0x + k0y + k0z) / 3.0
            kxmax = nx / 2
            kymax = ny / 2
            kzmax = nz / 2
            wave_numbers = knorm * np.arange(0, n)

            u, v, w = velocity[0, 0], velocity[0, 1], velocity[0, 2]

            uh = np.fft.fftn(u) / nt
            vh = np.fft.fftn(v) / nt
            wh = np.fft.fftn(w) / nt

            tkeh = 0.5 * (uh * np.conj(uh) + vh * np.conj(vh) + wh * np.conj(wh)).real

            tke_spectrum = np.zeros(len(wave_numbers))
            for kx in range(nx):
                rkx = kx if kx <= kxmax else kx - nx
                for ky in range(ny):
                    rky = ky if ky <= kymax else ky - ny
                    for kz in range(nz):
                        rkz = kz if kz <= kzmax else kz - nz
                        rk = np.sqrt(rkx**2 + rky**2 + rkz**2)
                        k = int(np.round(rk))
                        if k < len(tke_spectrum):
                            tke_spectrum[k] += tkeh[kx, ky, kz]

            tke_spectrum /= knorm
            tke_spectrum = tke_spectrum[1:]  # remove k=0
            if smooth:
                smoothed = movingaverage(tke_spectrum, 5)
                smoothed[0:4] = tke_spectrum[0:4]
                tke_spectrum = smoothed

            spectra.append(tke_spectrum)

    # Average over all snapshots
    tke_spectrum_avg = np.mean(spectra, axis=0)
    wave_numbers = wave_numbers[1:]

    # Kolmogorov analytical line
    C = 1.6          
    eps = 0.103 
    E_k_analytic = C * (eps ** (2 / 3)) * (wave_numbers ** (-5 / 3))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.loglog(wave_numbers, tke_spectrum_avg + 1e-20, label="Averaged TKE Spectrum")
    plt.loglog(wave_numbers, E_k_analytic, 'k--', label=r"$1.6 \, \varepsilon^{2/3} \, k^{-5/3}$")
    plt.xlabel("Wavenumber $k$")
    plt.ylabel("Energy $E(k)$")
    #plt.ylim(1e-7, 1)
    plt.title("Energy Spectrum")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("generated_plots", exist_ok=True)
    output_file = f"generated_plots/{name}.png"
    plt.savefig(output_file)
    plt.close()

    return tke_spectrum_avg

def residual_of_generated(dataset, samples, samples_gt, config):
    rmse_loss = np.zeros(len(samples))
    for i in range(len(samples)):
        # Ensure all tensors are on the same device
        sample = samples[i].unsqueeze(0).to(config.device)
        if dataset is not None:
            res = compute_divergence(sample[:, :3, :, :, :].to("cpu"), 2*math.pi/config.Data.grid_size)
        else:
            res = compute_divergence_fdm_wall_high_order(sample[:, :3, :, :, :].to("cpu"), hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
        rmse_loss[i] = torch.mean(torch.abs(res))
        #rmse_loss[i] = torch.sqrt(torch.sum(res**2))
    
    test_residuals = []
    for i in range(len(samples)):
        sample_gt = samples_gt[i].unsqueeze(0).to(config.device)
        if dataset is not None:
            res_gt = compute_divergence(sample_gt[:, :3, :, :, :].to("cpu"), 2*math.pi/config.Data.grid_size)
        else:
            res_gt = compute_divergence_fdm_wall_high_order(sample_gt[:, :3, :, :, :].to("cpu"), hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
        test_residuals.append(torch.mean(torch.abs(res_gt)))
        
    print(f"L2 residual: {np.mean(rmse_loss):.4f} +/- {np.std(rmse_loss):.4f} (max: {np.max(rmse_loss):.4f})") 
    # Ensure test_residuals is a numpy array on CPU
    test_residuals_np = np.array([r.cpu().item() if torch.is_tensor(r) else r for r in test_residuals])
    print(f"Residual difference: {np.mean(rmse_loss - test_residuals_np):.4f} +/- {np.std(rmse_loss - test_residuals_np):.4f} (max: {np.max(rmse_loss - test_residuals_np):.4f})")

    # Compute L2 norm of the difference between samples and ground truth
    l2_diff_norms = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample
        y_pred = samples[i]  # Retrieve saved y_pred
        l2_diff_norm = torch.sqrt(torch.mean((y - y_pred) ** 2)).item()
        l2_diff_norms.append(l2_diff_norm)

    print(f"Mean L2 difference between generated samples and ground truth: {np.mean(l2_diff_norms):.4f} +/- {np.std(l2_diff_norms):.4f} (max: {np.max(l2_diff_norms):.4f})")

def test_wasserstein(samples, samples_gt, config):
    wasserstein_cmf_distances = []
    for i in range(len(samples)):
        y = samples_gt[i] # Ground truth sample: (C, D, D, D)
        y_pred = samples[i].squeeze(0)  # Prediction: (C, D, D, D)
        
        wasserstein_cmf_distances.append(wasserstein(y, y_pred))

    # Mean and std
    mean_wasserstein = np.mean(wasserstein_cmf_distances)
    std_wasserstein = np.std(wasserstein_cmf_distances)
    print(f"Wasserstein distance: {mean_wasserstein:.4f} +/- {std_wasserstein:.4f} (max: {np.max(wasserstein_cmf_distances):.4f})")
    
def test_blurriness(samples, samples_gt, config):
    blurriness = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample: (C, D, D, D)
        y_pred = samples[i].squeeze(0)  # Prediction: (C, D, D, D)
        
        # Compute blurriness using Laplacian variance
        blurr_pred = compute_blurriness(y_pred.cpu().numpy())
        blurr_gt = compute_blurriness(y.cpu().numpy())
        blurriness.append(blurr_gt - blurr_pred)

    mean_blurriness = np.mean(blurriness)
    std_blurriness = np.std(blurriness)
    print(f"Sharpness: {mean_blurriness:.4f} +/- {std_blurriness:.4f} (max: {np.max(blurriness):.4f})")

def test_energy_spectrum(dataset, samples, samples_gt, config):
    spectrum = []
    samples_tensor = torch.stack([s.squeeze(0) for s in samples])
    for i in range(len(samples)):
        if dataset is not None:
            e_gt, k = compute_energy_spectrum(samples_gt[i].unsqueeze(0), f"energy_gt", config.device)
            e_fm, k = compute_energy_spectrum(samples_tensor[i].unsqueeze(0), f"energy_fm", config.device)
            
            compare_spectrums(e_gt, e_fm, k, f"energy_comp")
            
            k = torch.log10(k[:-15])
            diff = torch.abs(torch.log10(e_gt[:-15]) - torch.log10(e_fm[:-15]))
            
            k = k.cpu().numpy()
            diff = diff.cpu().numpy()
            area = np.trapz(diff, k)
            spectrum.append(area)
        else:
            wall_indices = list(range(0, 64))
            spectrum_tmp = 0
            for idx in wall_indices:
                u_fm = samples_tensor[i].unsqueeze(0)
                u_fm = u_fm[:, :, :, idx, :].squeeze(0)  # shape (3, 192, 192)
                u_gt = samples_gt[i].unsqueeze(0)
                u_gt = u_gt[:, :, :, idx, :].squeeze(0)
                nz_ = u_fm.shape[-1]
                kz_rfft = np.arange(1,nz_ / 2 + 0.1, 1)
                fft_fm = kz_rfft * np.mean(np.abs(np.fft.rfft(u_fm, axis=-1))**2, axis=(0,1))[:int(nz_ / 2)]    
                fft_gt = kz_rfft * np.mean(np.abs(np.fft.rfft(u_gt, axis=-1))**2, axis=(0,1))[:int(nz_ / 2)]   
                E_fm = np.max(fft_fm)
                E_gt = np.max(fft_gt)
                Energy_fm = fft_fm / E_fm
                Energy_gt = fft_gt / E_gt
                k = torch.log10(torch.from_numpy(kz_rfft))
                diff = torch.abs(torch.log10(torch.from_numpy(Energy_gt)) - torch.log10(torch.from_numpy(Energy_fm)))
                k = k[1:-1].cpu().numpy()
                diff = diff[1:-1].cpu().numpy()
                area = np.trapz(diff, k)
                spectrum_tmp += area
                
            spectrum.append(spectrum_tmp / len(wall_indices))
            
    print(f"Energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")

def S_function(x, y, c):
    num = 2 * x * y + c
    den = x**2 + y**2 + c
    return num / den

def empirical_covariance_3d(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    assert y.shape == y_hat.shape, "y and y_hat must have the same shape"
    B, C, Nx, Ny, Nz = y.shape
    N = Nx * Ny * Nz

    # Flatten spatial dimensions
    y_flat = y.view(C, -1)  # (C, N)
    y_hat_flat = y_hat.view(C, -1)  # (C, N)

    # Means
    mu_y = y_flat.mean(dim=1, keepdim=True)      # (C, 1)
    mu_yhat = y_hat_flat.mean(dim=1, keepdim=True)  # (C, 1)

    # Centered
    y_centered = y_flat - mu_y  # (C, N)
    yhat_centered = y_hat_flat - mu_yhat  # (C, N)

    # Covariance: sum over N then divide by (N - 1)
    cov = (y_centered * yhat_centered).sum(dim=1) / (N - 1)  # (C,)

    return cov
 
def ssim_function(sample, gt):
    N, C, Nx, Ny, Nz = sample.shape
    list_ssim = []
    
    corr = empirical_covariance_3d(sample, gt)
    
    for c in range(C):
        ch = sample[0, c, :, :, :]
        ch_gt = gt[0, c, :, :, :]
        
        mu1 = torch.mean(ch)
        std1 = torch.std(ch)
        mu2 = torch.mean(ch_gt)
        std2 = torch.std(ch_gt)
        
        S1 = S_function(mu1, mu2, (0.5 * ch_gt.max().item())**2)
        S2 = S_function(std1, std2, (0.5 * ch_gt.max().item())**2)
        
        coef = corr[c]
        c3 = 0.1
        S3 = (coef + c3) / (std1 * std2 + c3)
        
        return (S1 * S2 * S3).detach().cpu().numpy()
        
    
def experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, y_pred):
    losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
    
    mse = torch.mean((y_pred - y) ** 2).item()
    max_pixel = float(y.max().item())
    psnr_value = 20 * np.log10(max_pixel) - 10 * np.log10(mse + 1e-8)
    psnr.append(psnr_value)
    
    ssim.append(ssim_function(y_pred, y))
    
    if dataset is not None:
        residuals.append(torch.mean(torch.abs(compute_divergence(y_pred[:, :3, :, :, :].to("cpu"), 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(compute_divergence(y[:, :3, :, :, :].to("cpu"), 2*math.pi/config.Data.grid_size))).item())
    else:
        residuals.append(torch.mean(torch.abs(compute_divergence_fdm_wall_high_order(y_pred[:, :3, :, :, :].to("cpu"), hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192))).item())
        residuals_gt.append(torch.mean(torch.abs(compute_divergence_fdm_wall_high_order(y[:, :3, :, :, :].to("cpu"), hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192))).item())
    residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
    # Detach tensors before passing them to LSiM_distance
    y = y.detach()
    y_pred = y_pred.detach()
    lsim.append(LSiM_distance_3D(y, y_pred))
    
    y = y.squeeze(0)
    y_pred = y_pred.squeeze(0)
    blurr_pred = compute_blurriness(y_pred.cpu().numpy())
    blurr_gt = compute_blurriness(y.cpu().numpy())
    blurriness.append(blurr_gt - blurr_pred)
    
    y = y.unsqueeze(0)
    y_pred = y_pred.unsqueeze(0)
    if dataset is not None:
        e_gt, k = compute_energy_spectrum(y, f"energy_gt_{i}", config.device)
        e_pred, k = compute_energy_spectrum(y_pred, f"energy_pred_{i}", config.device)
        
        compare_spectrums(e_gt, e_pred, k, f"energy_comp_{i}")
        
        k = torch.log10(k[:-15])
        diff = torch.abs(torch.log10(e_gt[:-15]) - torch.log10(e_pred[:-15]))
        k = k.cpu().numpy()
        diff = diff.cpu().numpy()
        area = np.trapz(diff, k)
        spectrum.append(area)
    else:
        wall_indices = list(range(0, 64))
        spectrum_tmp = 0
        for idx in wall_indices:
            u_fm = y_pred
            u_fm = u_fm[:, :, :, idx, :].squeeze(0)  # shape (3, 192, 192)
            u_gt = y
            u_gt = u_gt[:, :, :, idx, :].squeeze(0)
            nz_ = u_fm.shape[-1]
            kz_rfft = np.arange(1,nz_ / 2 + 0.1, 1)
            fft_fm = kz_rfft * np.mean(np.abs(np.fft.rfft(u_fm.cpu().numpy(), axis=-1))**2, axis=(0,1))[:int(nz_ / 2)]  
            fft_gt = kz_rfft * np.mean(np.abs(np.fft.rfft(u_gt.cpu().numpy(), axis=-1))**2, axis=(0,1))[:int(nz_ / 2)] 
            E_fm = np.max(fft_fm)
            E_gt = np.max(fft_gt)
            Energy_fm = fft_fm / E_fm
            Energy_gt = fft_gt / E_gt
            k = torch.log10(torch.from_numpy(kz_rfft))
            diff = torch.abs(torch.log10(torch.from_numpy(Energy_gt)) - torch.log10(torch.from_numpy(Energy_fm)))
            k = k[1:-1].cpu().numpy()
            diff = diff[1:-1].cpu().numpy()
            area = np.trapz(diff, k)
            spectrum_tmp += area
            
        spectrum.append(spectrum_tmp / len(wall_indices))
    
    #diff = torch.abs((e_gt - e_pred) / e_gt) / k
    #diff = diff[:-15]
    #diff = torch.mean(diff)
    #spectrum.append(diff.cpu().numpy())

        
    
    
#def leray_projection(velocity, N):
#    device = velocity.device  # Get the device of the input tensor
#    #mesh_grid = MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi, N), (0, 2*torch.pi, N)], device=device)
#    mesh_grid=MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi * 5 / 1024, 5), (0, 2*torch.pi, N)], device=device)
#    div = Div()
#    divergence = div(velocity, mesh=mesh_grid)
#    
#    laplacian = Laplacian()
#    poisson_solution = laplacian.solve(b=divergence, mesh=mesh_grid, n_channel=1)
#    
#    grad_operator = Grad()
#    correction = grad_operator(poisson_solution, mesh=mesh_grid)
#    
#    velocity_new = velocity - correction
#    return velocity_new

def leray_projection(velocity, N):
    device = velocity.device  # Get the device of the input tensor
    mesh_grid = MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi, N), (0, 2*torch.pi, N)], device=device)
    #mesh_grid=MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi * 5 / 1024, 5), (0, 2*torch.pi, N)], device=device)
    
    leray_op = Leray()
    velocity_new = leray_op(u=velocity, mesh=mesh_grid)
    
    return velocity_new

def plot_q_isosurface(Q_tensor, velocity, q_threshold=0.1, vort_threshold=40, spacing=(1.0, 1.0, 1.0), cmap="viridis", save_path="q_isosurface.png"):
    
    vorticity = compute_vorticity(velocity, 2*math.pi / 128)
    vorticity_mag = torch.sqrt(torch.sum(vorticity**2, dim=1)).unsqueeze(0)
    vorticity_mag[vorticity_mag > vort_threshold] = vort_threshold
    vorticity_magnitude_tensor = vorticity_mag
    
    assert Q_tensor.shape[0] == 1 and Q_tensor.shape[1] == 1, "Q_tensor must be shape (1, 1, Nx, Ny, Nz)"
    assert vorticity_magnitude_tensor.shape == Q_tensor.shape, "Shape mismatch between Q and vorticity magnitude"

    # Remove batch/channel dimensions
    Q_np = Q_tensor[0, 0].detach().cpu().numpy()
    vort_mag_np = vorticity_magnitude_tensor[0, 0].detach().cpu().numpy()

    Nx, Ny, Nz = Q_np.shape
    dx, dy, dz = spacing

    # Create a PyVista uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = (Nx, Ny, Nz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (dx, dy, dz)

    # Add scalar fields to the grid
    grid["Q"] = Q_np.flatten(order="F")
    grid["vorticity_magnitude"] = vort_mag_np.flatten(order="F")

    # Extract the Q isosurface
    contours = grid.contour(isosurfaces=[q_threshold], scalars="Q")

    # Plotting
    plotter = pv.Plotter(off_screen=True)  # off_screen=True to enable saving without display
    plotter.add_mesh(contours, scalars="vorticity_magnitude", cmap=cmap, show_scalar_bar=False)
    plotter.hide_axes()
    plotter.add_mesh(grid.outline(), color="black", line_width=1.5)
    #plotter.add_title(f"Q Isosurface (Q = {q_threshold})", font_size=18)
    
    # Save the screenshot instead of showing interactively
    plotter.screenshot(save_path)
    plotter.close()
    
def plot_q_isosurface_ch(Q_tensor, velocity, q_threshold=0.1, vort_threshold=40, spacing=(1.0, 1.0, 1.0), cmap="viridis", save_path="q_isosurface.png"):
    """
    Plot a Q-criterion isosurface colored by vorticity magnitude and save as an image.

    Parameters:
        Q_tensor (torch.Tensor): Q-criterion field, shape (1, 1, Nx, Ny, Nz)
        vorticity_magnitude_tensor (torch.Tensor): Vorticity magnitude, shape (1, 1, Nx, Ny, Nz)
        q_threshold (float): Isosurface threshold for Q
        spacing (tuple): Grid spacing (dx, dy, dz)
        cmap (str): Colormap for vorticity magnitude
        save_path (str): File path to save the plot image (PNG format)
    """
    vorticity = compute_curl_fdm_wall_high_order(velocity, hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
    vorticity_mag = torch.sqrt(torch.sum(vorticity**2, dim=1)).unsqueeze(0)
    vorticity_mag[vorticity_mag > vort_threshold] = vort_threshold
    vorticity_magnitude_tensor = vorticity_mag
    
    assert Q_tensor.shape[0] == 1 and Q_tensor.shape[1] == 1, "Q_tensor must be shape (1, 1, Nx, Ny, Nz)"
    assert vorticity_magnitude_tensor.shape == Q_tensor.shape, "Shape mismatch between Q and vorticity magnitude"

    # Remove batch/channel dimensions
    Q_np = Q_tensor[0, 0].detach().cpu().numpy()
    vort_mag_np = vorticity_magnitude_tensor[0, 0].detach().cpu().numpy()

    Nx, Ny, Nz = Q_np.shape
    dx, dy, dz = spacing

    # Create a PyVista uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = (Nx, Ny, Nz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (dx, dy, dz)

    # Add scalar fields to the grid
    grid["Q"] = Q_np.flatten(order="F")
    grid["vorticity_magnitude"] = vort_mag_np.flatten(order="F")

    # Extract the Q isosurface
    contours = grid.contour(isosurfaces=[q_threshold], scalars="Q")
    contours = contours.smooth(n_iter=30, relaxation_factor=0.1, feature_smoothing=False)

    # Plotting
    plotter = pv.Plotter(off_screen=True)  # off_screen=True to enable saving without display
    plotter.add_mesh(contours, scalars="vorticity_magnitude", cmap=cmap, show_scalar_bar=False)
    plotter.hide_axes()
    plotter.add_mesh(grid.outline(), color="black", line_width=1.5)
    plotter.add_title(f"Q Isosurface (Q = {q_threshold})", font_size=12)
    
    # Save the screenshot instead of showing interactively
    plotter.screenshot(save_path, window_size=(6000, 4000))
    plotter.close()

def q_criterion(velocity, N):
    mesh_grid = MeshGrid([(0, 2*torch.pi, N), (0, 2*torch.pi, N), (0, 2*torch.pi, N)], device="cpu")
    velocity = velocity.to("cpu")
    
    grad_operator = Grad()
    nabla_u = grad_operator(velocity[:, 0, :, :, :].unsqueeze(0), mesh=mesh_grid)
    dudx = nabla_u[:, 0, :, :, :]
    dudy = nabla_u[:, 1, :, :, :]
    dudz = nabla_u[:, 2, :, :, :]
    
    nabla_v = grad_operator(velocity[:, 1, :, :, :].unsqueeze(0), mesh=mesh_grid)
    dvdx = nabla_v[:, 0, :, :, :]
    dvdy = nabla_v[:, 1, :, :, :]
    dvdz = nabla_v[:, 2, :, :, :]
    
    nabla_w = grad_operator(velocity[:, 2, :, :, :].unsqueeze(0), mesh=mesh_grid)
    dwdx = nabla_w[:, 0, :, :, :]
    dwdy = nabla_w[:, 1, :, :, :]
    dwdz = nabla_w[:, 2, :, :, :]
    
    S = 0.25 * ((dudx + dudx)**2 + (dudy + dvdx)**2 + (dudz + dwdx)**2 + 
                (dvdx + dudy)**2 + (dvdy + dvdy)**2 + (dvdz + dwdy)**2 +
                (dudz + dwdx)**2 + (dwdy + dvdz)**2 + (dwdz + dwdz)**2)
    
    O = 0.25 * ((dudx - dudx)**2 + (dudy - dvdx)**2 + (dudz - dwdx)**2 + 
                (dvdx - dudy)**2 + (dvdy - dvdy)**2 + (dvdz - dwdy)**2 +
                (dudz - dwdx)**2 + (dwdy - dvdz)**2 + (dwdz - dwdz)**2)
    
    Q = 0.5 * (O - S)
    
    return Q.unsqueeze(0)

def compute_dfdys(f, dy_tensor):
    B, Nx, Ny, Nz = f.shape
    dfdy = torch.zeros_like(f)

    # Interior points with 4th-order central difference
    for j in range(2, Ny - 2):
        h = (dy_tensor[j - 2] + dy_tensor[j - 1] + dy_tensor[j] + dy_tensor[j + 1]) / 4
        dfdy[:, :, j, :] = (
            -f[:, :, j + 2, :] + 8 * f[:, :, j + 1, :] - 8 * f[:, :, j - 1, :] + f[:, :, j - 2, :]
        ) / (12 * h)

    # Forward difference at bottom
    dfdy[:, :, 0, :] = (-3 * f[:, :, 0, :] + 4 * f[:, :, 1, :] - f[:, :, 2, :]) / (2 * (dy_tensor[0] + dy_tensor[1]) / 2)
    dfdy[:, :, 1, :] = (-3 * f[:, :, 1, :] + 4 * f[:, :, 2, :] - f[:, :, 3, :]) / (2 * (dy_tensor[1] + dy_tensor[2]) / 2)

    # Backward difference at top
    dfdy[:, :, -2, :] = (3 * f[:, :, -2, :] - 4 * f[:, :, -3, :] + f[:, :, -4, :]) / (2 * (dy_tensor[-3] + dy_tensor[-2]) / 2)
    dfdy[:, :, -1, :] = (3 * f[:, :, -1, :] - 4 * f[:, :, -2, :] + f[:, :, -3, :]) / (2 * (dy_tensor[-2] + dy_tensor[-1]) / 2)

    return dfdy

def q_criterion_ch(velocity, hx, Ny, hz):
    """
    Compute Q-criterion using 4th order finite differences with special y-boundary handling.
    velocity: tensor (batch, 3, Nx, Ny, Nz)
    hx, hy, hz: grid spacings
    """
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]
    
    y = yvector(Ny)
    y = y / 2
    dy = np.diff(y)[1:]
    dy_tensor = torch.from_numpy(dy).float().to(velocity.device)

    # Compute gradients
    dudx = fourth_order_diff(vx, 0, hx)
    dudy = compute_dfdys(vx, dy_tensor)
    dudz = fourth_order_diff(vx, 2, hz)

    dvdx = fourth_order_diff(vy, 0, hx)
    dvdy = compute_dfdys(vy, dy_tensor)
    dvdz = fourth_order_diff(vy, 2, hz)

    dwdx = fourth_order_diff(vz, 0, hx)
    dwdy = compute_dfdys(vz, dy_tensor)
    dwdz = fourth_order_diff(vz, 2, hz)

    S = 0.25 * ((dudx + dudx)**2 + (dudy + dvdx)**2 + (dudz + dwdx)**2 + 
                (dvdx + dudy)**2 + (dvdy + dvdy)**2 + (dvdz + dwdy)**2 +
                (dudz + dwdx)**2 + (dwdy + dvdz)**2 + (dwdz + dwdz)**2)
    O = 0.25 * ((dudx - dudx)**2 + (dudy - dvdx)**2 + (dudz - dwdx)**2 + 
                (dvdx - dudy)**2 + (dvdy - dvdy)**2 + (dvdz - dwdy)**2 +
                (dudz - dwdx)**2 + (dwdy - dvdz)**2 + (dwdz - dwdz)**2)
    Q = 0.5 * (O - S)
    
    return Q.unsqueeze(0)
    
def q_criterion_m(velocity, N):
    device = velocity.device
    mesh = MeshGrid([(0, 2*torch.pi, N)] * 3, device=device)
    grad = Grad()

    u = velocity[:, 0:1]  # Shape: [B, 1, N, N, N]
    v = velocity[:, 1:2]
    w = velocity[:, 2:3]

    du = grad(u, mesh=mesh)[0]  # [B, 3, N, N, N]
    dv = grad(v, mesh=mesh)[0]
    dw = grad(w, mesh=mesh)[0]

    # grad_tensor[i, j] = ∂u_i / ∂x_j
    grad_tensor = torch.stack([du, dv, dw], dim=1)  # [B, 3, 3, N, N, N]

    # Transpose i <-> j
    grad_T = grad_tensor.transpose(0, 1)  # [B, 3, 3, N, N, N]

    S = 0.5 * (grad_tensor + grad_T)
    Omega = 0.5 * (grad_tensor - grad_T)

    S_norm2 = (S ** 2).sum(dim=(0, 1))       # [B, N, N, N]
    Omega_norm2 = (Omega ** 2).sum(dim=(0, 1))

    Q = 0.5 * (Omega_norm2 - S_norm2)        # [B, N, N, N]
    
    return Q.unsqueeze(0).unsqueeze(1)                    # [B, 1, N, N, N]


def collect_n_samples(test_loader, n):
    collected = []
    total = 0
    
    for batch in test_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch  # Handle (input, target)
        
        for i in range(x.size(0)):
            collected.append(x[i].unsqueeze(0))  # Keep batch dimension
            total += 1
            if total >= n:
                break
        if total >= n:
            break

    collected_tensor = torch.cat(collected, dim=0)
    return collected_tensor

def collect_n_samples_supervised(test_loader, n):
    lr_list = []
    hr_list = []
    count = 0

    for batch_lr, batch_hr in test_loader:
        for i in range(batch_lr.shape[0]):
            lr_list.append(batch_lr[i])
            hr_list.append(batch_hr[i])
            count += 1
            if count >= n:
                break
        if count >= n:
            break

    test_lr = torch.stack(lr_list, dim=0)
    test_hr = torch.stack(hr_list, dim=0)

    return test_lr, test_hr


def experiment_ch_y(config, y, y_pred):
    assert y.shape == y_pred.shape, "y and y_pred must have the same shape"
    assert y.dim() == 5 and y.shape[0] == 1, "Expected shape (1, 3, Nx, Ny, Nz)"

    _, _, Nx, Ny, Nz = y.shape
    rmse_per_y = torch.zeros(Ny, device=y.device)
    lsim_per_y = torch.zeros(Ny, device=y.device)
    spectrum_per_y = torch.zeros(Ny, device=y.device)

    for j in range(Ny):
        gt_slice = y[0, :, :, j, :]       # shape (3, Nx, Nz)
        pred_slice = y_pred[0, :, :, j, :]  # shape (3, Nx, Nz)
        mse = F.mse_loss(pred_slice, gt_slice, reduction='mean')
        rmse_per_y[j] = torch.sqrt(mse)
        
        #lsim_per_y[j] = torch.tensor(LSiM_distance(gt_slice, pred_slice), device=lsim_per_y.device)
        
        #e_gt, k = compute_energy_spectrum_2D(gt_slice.unsqueeze(0), f"energy_gt", config.device, lx=2*math.pi, ly=math.pi)
        #e_pred, k = compute_energy_spectrum_2D(pred_slice.unsqueeze(0), f"energy_pred", config.device, lx=2*math.pi, ly=math.pi)
        #compare_spectrums(e_gt, e_pred, k, f"energy_comp")
        #k = torch.log10(k[:90])
        #diff = torch.abs(torch.log10(e_gt[:90]) - torch.log10(e_pred[:90]))
        #k = k.cpu().numpy()
        #diff = diff.cpu().numpy()
        #area = np.trapz(diff, k)
        #spectrum_per_y[j] = area
        
    return rmse_per_y#, lsim_per_y, spectrum_per_y

def plot_report_8_ch(dataset, config, indices, samples_x, samples_y, reg, ddpm, fm, dp, cond, hybrid, filename):
    n = len(indices)
    #fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 3 * n))
    fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 1.5 * n))

    for row, idx in enumerate(indices):
        
        # Low-res
        axes[row, 0].imshow(
            #samples_x[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            samples_x[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 0].axis('off')
        
        # Regression
        axes[row, 1].imshow(
            #reg[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            reg[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 1].axis('off')
        
        # DDPM
        axes[row, 2].imshow(
            #ddpm[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            ddpm[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 2].axis('off')
        
        # FM
        axes[row, 3].imshow(
            #fm[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            fm[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 3].axis('off')
        
        # DP
        axes[row, 4].imshow(
            #dp[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            dp[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 4].axis('off')
        
        # Cond
        axes[row, 5].imshow(
            #cond[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            cond[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 5].axis('off')
        
        # Hybrid
        axes[row, 6].imshow(
            #hybrid[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            hybrid[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 6].axis('off')
        
        # Ground truth
        im = axes[row, 7].imshow(
            #samples_y[idx, 0, :, :, int(config.Data.Nz / 2)].cpu().detach().numpy(),
            samples_y[idx, 0, int(config.Data.Nx / 2), :, :].cpu().detach().numpy(),
            cmap="twilight",
        )
        axes[row, 7].axis('off')
        
        # Lave the fourth subplot blank but place the colorbar there
        #axes[row, 7].axis('off')
        #cbar = fig.colorbar(im, ax=axes[row, 7], fraction=1.0)
        #cbar.ax.tick_params(labelsize=10)

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation", fontsize=16)
    axes[0, 1].set_title("P3D baseline", fontsize=16)
    axes[0, 2].set_title("DDPM mask method", fontsize=16)
    axes[0, 3].set_title("FM interp method", fontsize=16)
    axes[0, 4].set_title("FM DP method", fontsize=16)
    axes[0, 5].set_title("FM cond method", fontsize=16)
    axes[0, 6].set_title("Hybrid method", fontsize=16)
    axes[0, 7].set_title("Reference", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    
    compare_spectrum_7_ch(reg, ddpm, fm, dp, cond, hybrid, samples_y, filename)
    
def plot_report_8_3d_ch(dataset, config, indices, samples_x, samples_y, reg, ddpm, fm, dp, cond, hybrid, filename, norm=False, fluctuation=False):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 3 * n))
    
    if norm:
        samples_x = dataset.data_scaler(samples_x)
        samples_y = dataset.data_scaler(samples_y)
        reg = dataset.data_scaler(reg)
        ddpm = dataset.data_scaler(ddpm)
        fm = dataset.data_scaler(fm)
        dp = dataset.data_scaler(dp)
        hybrid = dataset.data_scaler(hybrid)
        cond = dataset.data_scaler(cond)
        
    if fluctuation:
        samples_x = samples_x - dataset.mean_velocity_field
        samples_y = samples_y - dataset.mean_velocity_field
        reg = reg - dataset.mean_velocity_field
        ddpm = ddpm - dataset.mean_velocity_field
        fm = fm - dataset.mean_velocity_field
        dp = dp - dataset.mean_velocity_field
        cond = cond - dataset.mean_velocity_field
        hybrid = hybrid - dataset.mean_velocity_field
        
    
    base_cmap = plt.get_cmap("turbo")
    listed_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    colormap = ramp_step_alpha(listed_cmap, 0.2, 0.99)

    for row, idx in enumerate(indices):
        # Low-res
        velocity = samples_x[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img0 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 0].imshow(img0)
        axes[row, 0].axis("off")
        
        # Regression
        velocity = reg[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img1 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 1].imshow(img1)
        axes[row, 1].axis("off")
        
        # DDPM Mask
        velocity = ddpm[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img2 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 2].imshow(img2)
        axes[row, 2].axis("off")
        
        # FM interp
        velocity = fm[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img3 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 3].imshow(img3)
        axes[row, 3].axis("off")
        
        # FM dp
        velocity = dp[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img4 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 4].imshow(img4)
        axes[row, 4].axis("off")
        
        # FM cond
        velocity = cond[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img5 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 5].imshow(img5)
        axes[row, 5].axis("off")
        
        # Hybrid
        velocity = hybrid[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img6 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 6].imshow(img6)
        axes[row, 6].axis("off")
        
        # Ground truth
        velocity = samples_y[idx].unsqueeze(0)
        if dataset is not None:
            q = q_criterion(velocity, 128)
        else:
            q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
        q = q.cpu().numpy()
        img7 = render(
            q[0],
            cmap=colormap,
            background=(0, 0, 0, 1),
            vmin=0,
            vmax=500,
            distance_scale=4.0,
        )
        axes[row, 7].imshow(img7)
        axes[row, 7].axis("off")
        
        np.save("data/q.npy", q)
        print(q.min(), q.max())

    # Add column titles to the first row
    axes[0, 0].set_title("Interpolation")
    axes[0, 1].set_title("P3D baseline")
    axes[0, 2].set_title("DDPM mask method")
    axes[0, 3].set_title("FM interp method")
    axes[0, 4].set_title("FM DP method")
    axes[0, 5].set_title("FM cond method")
    axes[0, 6].set_title("Hybrid method")
    axes[0, 7].set_title("Reference")

    plt.tight_layout()
    plt.savefig(f"generated_plots/{filename}.png", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Plot saved in generated_plots/{filename}.png")
    

def plot_report_8_3d_iso_ch(dataset, config, indices, samples_x, samples_y, reg, ddpm, fm, dp, cond, hybrid, filename, norm=False, fluctuation=False, q_threshold=150):
    os.makedirs("generated_plots/frames", exist_ok=True)
    
    qs_interp = []
    qs_base = []
    qs_ddpm = []
    qs_fm = []
    qs_dp = []
    qs_cond = []
    qs_hybrid = []
    qs_ref = []
    
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=8, figsize=(18, 1.5 * n))

    for row, idx in enumerate(indices):
        entries = [
            ("Interpolation", samples_x[idx]),
            ("P3D baseline", reg[idx]),
            ("DDPM mask method", ddpm[idx]),
            ("FM interp method", fm[idx]),
            ("FM DP method", dp[idx]),
            ("FM cond method", cond[idx]),
            ("Hybrid method", hybrid[idx]),
            ("Reference", samples_y[idx]),
        ]
        
        for col, (title, velocity_tensor) in enumerate(entries):
            velocity = velocity_tensor.unsqueeze(0)
            if dataset is not None:
                q = q_criterion(velocity, 128)
            else:
                q = q_criterion_ch(velocity, hx=2*math.pi / 192, Ny=65, hz=math.pi/192)
            
            q_flat = q[0, 0].detach().cpu().numpy().flatten()
            if col == 0:
                qs_interp.append(q_flat)
            if col == 1:
                qs_base.append(q_flat)
            if col == 2:
                qs_ddpm.append(q_flat)
            if col == 3:
                qs_fm.append(q_flat)
            if col == 4:
                qs_dp.append(q_flat)
            if col == 5:
                qs_cond.append(q_flat)
            if col == 6:
                qs_hybrid.append(q_flat)
            if col == 7:
                qs_ref.append(q_flat)
            
            save_path = f"generated_plots/frames/frame_{filename}_row{row}_col{col}.png"
            
            if dataset is not None:
                plot_q_isosurface(
                    Q_tensor=q,
                    velocity=velocity,
                    q_threshold=120,
                    vort_threshold=35,
                    spacing=(2*np.pi/128, 2*np.pi/128, 2*np.pi/128),
                    cmap="jet",
                    save_path=save_path,
                )
            else:
                plot_q_isosurface_ch(
                    Q_tensor=q,
                    velocity=velocity,
                    vort_threshold=25,
                    q_threshold=15,
                    spacing=(2*np.pi/192, 1/65, np.pi/192),
                    cmap="jet",
                    save_path=save_path,
                )
            
            img = plt.imread(save_path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            
            if row == 0:
                axes[row, col].set_title(title, fontsize=16)
                
        np.save("data/q.npy", q.cpu().numpy())
        print(f"Row {row}: Q min {q.min().item()}, max {q.max().item()}")

    plt.tight_layout()
    final_path = f"generated_plots/{filename}.png"
    plt.savefig(final_path, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"Plot saved in {final_path}")
    
    
    method_qs = {
        "Interpolation": np.concatenate(qs_interp),
        "P3D baseline": np.concatenate(qs_base),
        "Mask Diffusion": np.concatenate(qs_ddpm),
        "FM interp": np.concatenate(qs_fm),
        "Direct path": np.concatenate(qs_dp),
        "Conditioning": np.concatenate(qs_cond),
        "Hybrid": np.concatenate(qs_hybrid),
        "Reference": np.concatenate(qs_ref),
    }

    plt.figure(figsize=(10, 6))

    # Define x-range for evaluation
    x_values = np.linspace(-25, 25, 500)

    for label, q_values in method_qs.items():
        q_pos = q_values
        kde = gaussian_kde(q_pos)
        density = kde(x_values)
        plt.plot(x_values, density, label=label, linewidth=2)

    plt.xlabel("Q value", fontsize=26)
    plt.ylabel("Probability Density", fontsize=26)
    plt.title("Q-criterion Distribution per Method", fontsize=26)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    dist_path = f"generated_plots/ch_q_distribution_kde_{filename}.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"Q-distribution KDE plot saved in {dist_path}")