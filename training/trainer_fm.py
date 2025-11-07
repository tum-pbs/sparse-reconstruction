import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import wandb
from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator
import math
import random

from torchfsm.mesh import MeshGrid
from torchfsm.operator import Div
from dataset import BigSpectralIsotropicTurbulenceDataset
import utils
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L
from my_config_length import UniProjectionLength
import turb_datasets

def fm_standard_step(dataset, model, xt, t, target, optimizer, config, accumulation_steps, batch_idx, length):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean() / accumulation_steps
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
    if config.Data.case == "iso":
        eq_residual = utils.compute_divergence(dataset.data_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
    elif config.Data.case == "channel":
        eq_residual = utils.compute_divergence_fdm_wall_high_order(x1_pred[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
    eq_res_m = torch.mean(torch.abs(eq_residual)) / accumulation_steps
    
    total_loss = loss
    
    total_loss.backward() 
    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == length:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss, eq_res_m

def fm_PINN_step(dataset, model, xt, t, target, optimizer, config, accumulation_steps, batch_idx, length):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean() / accumulation_steps
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
    if config.Data.case == "iso":
        eq_residual = utils.compute_divergence(dataset.data_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
    elif config.Data.case == "channel":
        eq_residual = utils.compute_divergence_fdm_wall_high_order(x1_pred[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
    eq_res_m = torch.mean(torch.abs(eq_residual)) / accumulation_steps

    # Combine the flow matching loss and the divergence-free loss
    total_loss = loss + config.Training.divergence_loss_weight * eq_res_m
    
    total_loss.backward() 
    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == length:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss, eq_res_m

def fm_PINN_dyn_step(dataset, model, x1, xt, t, target, optimizer, config, accumulation_steps, batch_idx, length):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean() / accumulation_steps
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
    if config.Data.case == "iso":
        eq_residual = utils.compute_divergence(dataset.data_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
        eq_res_m = torch.mean(torch.abs(eq_residual))
    elif config.Data.case == "channel":
        eq_residual = utils.compute_divergence_fdm_wall_high_order(x1_pred[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
        eq_residual_gt = utils.compute_divergence_fdm_wall_high_order(x1[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
        eq_diff = eq_residual - eq_residual_gt
        eq_res_m = torch.mean(torch.abs(eq_diff))
        
    eq_res_m = eq_res_m / accumulation_steps


    # Combine the flow matching loss and the divergence-free loss
    coef = loss / eq_res_m
    total_loss = loss + coef * eq_res_m
    
    total_loss.backward() 
    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == length:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss, eq_res_m

def fm_ConFIG_step(dataset, model, x1, xt, t, target, optimizer, config, operator, accumulation_steps, batch_idx, length, grads1, grads2):
    optimizer.zero_grad()
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean() / accumulation_steps
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
    if config.Data.case == "iso":
        eq_residual = utils.compute_divergence(dataset.data_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
        eq_res_m = torch.mean(torch.abs(eq_residual))
    elif config.Data.case == "channel":
        eq_residual = utils.compute_divergence_fdm_wall_high_order(x1_pred[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
        eq_residual_gt = utils.compute_divergence_fdm_wall_high_order(x1[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
        eq_diff = eq_residual - eq_residual_gt
        eq_res_m = torch.mean(torch.abs(eq_diff))
        
    eq_res_m = eq_res_m / accumulation_steps
    
    # ConFIG
    loss_physics_unscaled = eq_res_m.clone()
    loss.backward(retain_graph=True)
    grads1.append(get_gradient_vector(model, none_grad_mode="skip"))
    optimizer.zero_grad()
    eq_res_m.backward()
    grads2.append(get_gradient_vector(model, none_grad_mode="skip"))

    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == length:
        grad_1 = sum(grads1) 
        grad_2 = sum(grads2) 
        operator.update_gradient(model, [grad_1, grad_2])
        #apply_gradient_vector(model, grad_1 + grad_2)
        #print(grad_1 + grad_2)
        grads1.clear()
        grads2.clear()
        optimizer.step()
        optimizer.zero_grad()

    return loss, loss_physics_unscaled

# Define the training function
def train_flow_matching(config):
    # Load the dataset
    print("Loading dataset...")

    if config.Data.case == "iso":
        dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config.Data.grid_size,
                                                        norm=config.Data.norm,
                                                        size=config.Data.size,
                                                        train_ratio=0.8,
                                                        val_ratio=0.1,
                                                        test_ratio=0.1,
                                                        batch_size=config.Training.batch_size,
                                                        num_samples=10)

        train_loader = dataset.train_loader
        val_loader = dataset.val_loader
        test_loader = dataset.test_loader
    
    elif config.Data.case == "channel":
        train_loader, val_loader, test_loader = turb_datasets.load_data(dataset_path="/mnt/data2/luca/channel", batch_size=config.Training.batch_size, class_cond=False)
        dataset = None

    # Initialize the model
    model = PDEDiT3D_B(
        channel_size=config.Model.channel_size,
        channel_size_out=config.Model.channel_size_out,
        drop_class_labels=config.Model.drop_class_labels,
        partition_size=config.Model.partition_size,
        mending=False
    )
    model = model.to(config.device)

    # Convert learning_rate and divergence_loss_weight to float if they are strings
    if isinstance(config.Training.learning_rate, str):
        config.Training.learning_rate = float(config.Training.learning_rate)
    if isinstance(config.Training.divergence_loss_weight, str):
        config.Training.divergence_loss_weight = float(config.Training.divergence_loss_weight)
    if isinstance(config.Training.sigma_min, str):
        config.Training.sigma_min = float(config.Training.sigma_min)
    if isinstance(config.Training.gamma, str):
        config.Training.gamma = float(config.Training.gamma)
    if isinstance(config.Training.last_lr, str):
        config.Training.last_lr = float(config.Training.last_lr)

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.Training.learning_rate)

    # Find the next run directory
    runs_dir = f'/mnt/data1/navarro/runs'
    existing = [d for d in os.listdir(runs_dir) if d.isdigit()]
    if existing:
        next_run = f"{max([int(d) for d in existing])+1:03d}"
    else:
        next_run = "001"
    run_dir = os.path.join(runs_dir, next_run)
    os.makedirs(run_dir, exist_ok=True)

    # Training loop with validation loss
    print("Starting training...")
    mse_losses = []
    val_losses = []
    accumulation_steps = config.Training.effective_batch_size // config.Training.batch_size
    optimizer.zero_grad()
    operator = ConFIGOperator(length_model=UniProjectionLength())
    for epoch in range(config.Training.epochs):
        model.train()
        mse_loss = 0.0

        # Get the next batch from the train_loader
        grads1 = []
        grads2 = []
        for batch_idx, x1 in enumerate(train_loader):
            #print(f"Batch {batch_idx+1}/{len(train_loader)}")
            if dataset is None:
                x1 = x1[0]
                x1 = x1[:, :, :, 1:, :]
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Ensure all elements in the batch are tensors
            x1 = torch.tensor(x1) if isinstance(x1, np.ndarray) else x1
            x0 = torch.randn_like(x1)
            target = x1 - (1 - config.Training.sigma_min) * x0

            x1 = x1.to(config.device)
            x0 = x0.to(config.device)
            target = target.to(config.device)

            # Sample random time steps
            t = torch.rand(x1.size(0), device=config.device)

            # Interpolate between x0 and x1
            xt = (1 - (1 - config.Training.sigma_min) * t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1
            xt = xt.float()
            
            # Perform the training step
            if config.Training.method == "std":
                loss, physics_loss = fm_standard_step(dataset, model, xt, t, target, optimizer, config, accumulation_steps, batch_idx, len(train_loader))
                
            elif config.Training.method == "PINN":
                loss, physics_loss = fm_PINN_step(dataset, model, xt, t, target, optimizer, config, accumulation_steps, batch_idx, len(train_loader))
                
            elif config.Training.method == "PINN_dyn":
                loss, physics_loss = fm_PINN_dyn_step(dataset, model, x1, xt, t, target, optimizer, config, accumulation_steps, batch_idx, len(train_loader))
                
            elif config.Training.method == "ConFIG":
                loss, physics_loss = fm_ConFIG_step(dataset, model, x1, xt, t, target, optimizer, config, operator, accumulation_steps, batch_idx, len(train_loader), grads1, grads2)
                
            else:
                raise ValueError(f"Unknown training method: {config.Training.method}")
            
            mse_loss += loss.item()
            
        mse_loss /= (len(train_loader) / accumulation_steps)
        mse_losses.append(mse_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        divergence_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                if dataset is None:
                    val_batch = val_batch[0]
                    val_batch = val_batch[:, :, :, 1:, :]
                    
                x1 = val_batch
                x0 = torch.randn_like(x1)
                target = x1 - x0

                # Ensure all tensors are on the same device
                x1 = x1.to(config.device)
                x0 = x0.to(config.device)
                target = target.to(config.device)

                t = torch.rand(x1.size(0), device=config.device)
                xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1

                pred = model(xt, t)
                pred = pred.sample
                val_loss += ((target - pred) ** 2).mean().item()
                
                x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
                if config.Data.case == "iso":
                    eq_residual = utils.compute_divergence(dataset.data_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
                elif config.Data.case == "channel":
                    eq_residual = utils.compute_divergence_fdm_wall_high_order(x1_pred[:, :3, :, :, :], hx=2 * math.pi / 192, Ny=65, hz = math.pi / 192)
                
                eq_res_m = torch.mean(torch.abs(eq_residual))
                divergence_loss += eq_res_m

        val_loss /= len(val_loader)
        divergence_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": mse_loss,
            "validation_loss": val_loss,
            "divergence_loss": divergence_loss
        })
        
        # Custom LR scheduler: multiply by gamma, but do not go below last_lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * config.Training.gamma, config.Training.last_lr)
            param_group['lr'] = new_lr

        # Save checkpoint every 100 epochs
        if config.Data.case == "iso":
            epochs_period = 100
        elif config.Data.case == "channel":
            epochs_period = 10
        if (epoch + 1) % epochs_period == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}_{mse_loss:.4f}_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Log the epoch loss and validation loss
        print(f"Epoch [{epoch + 1}/{config.Training.epochs}], Loss: {mse_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation divergence: {divergence_loss:.4f}")

    # Plot losses after training
    plt.figure()
    plt.plot(range(1, config.Training.epochs + 1), mse_losses, label='Train MSE Loss')
    plt.plot(range(1, config.Training.epochs + 1), val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    image_path = os.path.join(run_dir, "history.png")
    plt.savefig(image_path)

def start_training():
    # Load the configuration
    print("Loading config...")
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)

    # Train the model
    train_flow_matching(config)