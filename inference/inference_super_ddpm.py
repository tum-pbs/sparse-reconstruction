import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import utils
import random
import time

from dataset import BigSpectralIsotropicTurbulenceDataset
import utils
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L
from diffusion import Diffusion

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Linear Beta Schedule (from beta_min to beta_max over the T timesteps)
def get_linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod  

def ddim_interp(model, x, x_lr, t_start, reverse_steps, betas, alphas_cumprod):
    seq = range(0, t_start, t_start // reverse_steps)
    next_seq = [-1] + list(seq[:-1])
    n = x.size(0)  # Batch size

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            #print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

            alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
            alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t
            
            e = model(x, t)
            e = e.sample

            # Classic DDIM x0 prediction and update
            x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
            
            x_interp = x0_pred * (1 - t / t_start) + x_lr * (t / t_start)
            
            x = alpha_bar_next.sqrt() * x_interp + (1 - alpha_bar_next).sqrt() * e

            # Free memory of intermediate tensors
            del e, x0_pred
            torch.cuda.empty_cache()

    return x

def ddpm_shu_sparse_experiment(dataset, config, diffusion, model, nsamples, samples_x, samples_y, t_start=1000, reverse_steps=20, T=1000, leray=False):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    spectrum = []
    pred = []
    samples_in = []
    samples_out = []
    psnr = []
    ssim = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        
        start_time = time.time()
        y_pred = diffusion.ddim_article(x, model, t_start, reverse_steps, K=1)
        eval_time = time.time() - start_time
        print(f"Evaluation time: {eval_time:.4f} seconds")
        
        if dataset.norm:
            x = dataset.data_scaler.inverse(x)
            y = dataset.data_scaler.inverse(y)
            y_pred = dataset.data_scaler.inverse(y_pred)
        if leray:
            iter = 20
            for j in range(iter):
                y_pred = utils.leray_projection(y_pred, config.Data.grid_size)  
        samples_in.append(x)
        samples_out.append(y)
        pred.append(y_pred)  
        #utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         f"super_shu_ddpm_{i}")

        utils.experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, y_pred)
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"PSNR: {np.mean(psnr):.4f} +/- {np.std(psnr):.4f} (max: {np.max(psnr):.4f})")
    print(f"SSIM: {np.mean(ssim):.4f} +/- {np.std(ssim):.4f} (max: {np.max(ssim):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")
    
    return torch.stack(pred, dim=0).squeeze(1), torch.stack(samples_in, dim=0).squeeze(1), torch.stack(samples_out, dim=0).squeeze(1)
    
def ddpm_interp_sparse_experiment(dataset, config, diffusion, model, nsamples, samples_x, samples_y, t_start=1000, reverse_steps=20, T=1000, leray=False):
    betas, alphas_cumprod = get_linear_beta_schedule(config.Diffusion.num_diffusion_timesteps, config.Diffusion.beta_start, config.Diffusion.beta_end)
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    spectrum = []
    pred = []
    samples_in = []
    samples_out = []
    psnr = []
    ssim = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()
        
        y_pred = ddim_interp(model, noise.clone(), x.clone(), t_start, reverse_steps, betas, alphas_cumprod)
        if dataset.norm:
            x = dataset.data_scaler.inverse(x)
            y = dataset.data_scaler.inverse(y)
            y_pred = dataset.data_scaler.inverse(y_pred)
        if leray:
            iter = 20
            for j in range(iter):
                y_pred = utils.leray_projection(y_pred, config.Data.grid_size) 
        samples_in.append(x)
        samples_out.append(y)
        pred.append(y_pred)  
        #utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         f"super_interp_ddpm_{i}")

        utils.experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, y_pred)
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"PSNR: {np.mean(psnr):.4f} +/- {np.std(psnr):.4f} (max: {np.max(psnr):.4f})")
    print(f"SSIM: {np.mean(ssim):.4f} +/- {np.std(ssim):.4f} (max: {np.max(ssim):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")
    
    return torch.stack(pred, dim=0).squeeze(1), torch.stack(samples_in, dim=0).squeeze(1), torch.stack(samples_out, dim=0).squeeze(1)
    
def ddpm_mask_sparse_experiment(dataset, config, diffusion, model, nsamples, samples_x, samples_y, samples_ids, w_mask=0.0, t_start=1000, reverse_steps=20, T=1000, leray=False):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    spectrum = []
    pred = []
    samples_in = []
    samples_out = []
    psnr = []
    ssim = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        if samples_ids is not None:
            mask = torch.zeros(config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).flatten()
            mask[samples_ids[i]] = 1
            mask = mask.reshape(config.Data.grid_size, config.Data.grid_size, config.Data.grid_size)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, D, D)
            mask = mask.repeat(1, config.Model.channel_size, 1, 1, 1)  # (1, C, D, D, D)
            mask = mask.to(config.device)

        else:
            mask = None

        y_pred = diffusion.ddim_mask(noise.clone(), model, x.clone(), t_start, reverse_steps, w_mask=w_mask, _mask=mask)
        
        if dataset.norm:
            x = dataset.data_scaler.inverse(x)
            y = dataset.data_scaler.inverse(y)
            y_pred = dataset.data_scaler.inverse(y_pred)
        if leray:
            iter = 20
            for j in range(iter):
                y_pred = utils.leray_projection(y_pred, config.Data.grid_size) 
        samples_in.append(x)
        samples_out.append(y)
        pred.append(y_pred)   
        #utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         f"super_mask_ddpm_{i}")

        utils.experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, y_pred)
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"PSNR: {np.mean(psnr):.4f} +/- {np.std(psnr):.4f} (max: {np.max(psnr):.4f})")
    print(f"SSIM: {np.mean(ssim):.4f} +/- {np.std(ssim):.4f} (max: {np.max(ssim):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")
    
    return torch.stack(pred, dim=0).squeeze(1), torch.stack(samples_in, dim=0).squeeze(1), torch.stack(samples_out, dim=0).squeeze(1)
    
def ddpm_diff_mask_sparse_experiment(dataset, config, diffusion, model, nsamples, samples_x, samples_y, samples_ids, w_mask=0.0, sig=0.044, t_start=1000, reverse_steps=20, T=1000, leray=False):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    spectrum = []
    pred = []
    samples_in = []
    samples_out = []
    psnr = []
    ssim = []
    
    if samples_ids is not None:
        diffuse_masks = torch.zeros(len(samples_ids), config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).to(config.device)
        for j in range(len(samples_ids)):
            # Use the correct number of total voxels for 3D
            total_voxels = config.Data.grid_size ** 3
            ids = list(samples_ids[j])
            mask = utils.diffuse_mask(
                ids, A=1, sig=sig,
                Nx=config.Data.grid_size,
                Ny=config.Data.grid_size,
                Nz=config.Data.grid_size
            )
            diffuse_masks[j] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(config.Model.channel_size, 1, 1, 1)
    else:
        diffuse_masks = torch.zeros(nsamples, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).to(config.device)
        for j in range(nsamples):
            total_voxels = config.Data.grid_size ** 3
            ids = random.sample(range(total_voxels), int(total_voxels * w_mask))
            mask = utils.diffuse_mask(
                ids, A=1, sig=sig, 
                Nx=config.Data.grid_size,
                Ny=config.Data.grid_size,
                Nz=config.Data.grid_size
            )
            diffuse_masks[j] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(config.Model.channel_size, 1, 1, 1)
    
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        start_time = time.time()
        y_pred = diffusion.ddim_mask(noise.clone(), model, x.clone(), t_start, reverse_steps, diff_mask=diffuse_masks[i].unsqueeze(0), w_mask=w_mask)
        
        eval_time = time.time() - start_time
        print(f"Evaluation time: {eval_time:.4f} seconds")
        
        if dataset.norm:
            x = dataset.data_scaler.inverse(x)
            y = dataset.data_scaler.inverse(y)
            y_pred = dataset.data_scaler.inverse(y_pred)
        if leray:
            iter = 20
            for j in range(iter):
                y_pred = utils.leray_projection(y_pred, config.Data.grid_size)  
        samples_in.append(x)
        samples_out.append(y)
        pred.append(y_pred)  
        #utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
        #                         f"super_diff_mask_ddpm_{i}")

        utils.experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, y_pred)
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"PSNR: {np.mean(psnr):.4f} +/- {np.std(psnr):.4f} (max: {np.max(psnr):.4f})")
    print(f"SSIM: {np.mean(ssim):.4f} +/- {np.std(ssim):.4f} (max: {np.max(ssim):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")

    return torch.stack(pred, dim=0).squeeze(1), torch.stack(samples_in, dim=0).squeeze(1), torch.stack(samples_out, dim=0).squeeze(1)

# Main script
if __name__ == "__main__":
    print("Loading config...")
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    # Generate samples using ODE integration
    num_samples = 10
    
    dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config.Data.grid_size,
                                                    norm=config.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=num_samples)
    samples_gt = dataset.test_dataset
    
    # Load the trained model
    print("Loading model...")
    model = PDEDiT3D_B(
        channel_size=config.Model.channel_size,
        channel_size_out=config.Model.channel_size_out,
        drop_class_labels=config.Model.drop_class_labels,
        partition_size=config.Model.partition_size,
        mending=False
    )
    model.load_state_dict(torch.load(config.Model.save_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    samples_y = dataset.test_dataset
    
    #perc = 5
    #samples_x, samples_ids = utils.interpolate_dataset(samples_y, perc/100)
    samples_x, samples_ids = utils.downscale_data(samples_y, 4)
    
    # Diffusion parameters
    diffusion = Diffusion(config)
    
    pred_shu, samples_x_norm, samples_y_norm = ddpm_shu_sparse_experiment(dataset, config, diffusion, model, num_samples, samples_x, samples_y, reverse_steps=100, leray=False)
    pred_interp, samples_x_norm, samples_y_norm = ddpm_interp_sparse_experiment(dataset, config, diffusion, model, num_samples, samples_x, samples_y, reverse_steps=100, leray=False)
    pred_mask, samples_x_norm, samples_y_norm = ddpm_mask_sparse_experiment(dataset, config, diffusion, model, num_samples, samples_x, samples_y, samples_ids, reverse_steps=100, leray=False, w_mask=0.0)
    pred_diff_mask, samples_x_norm, samples_y_norm = ddpm_diff_mask_sparse_experiment(dataset, config, diffusion, model, num_samples, samples_x, samples_y, samples_ids,
                                                                                      reverse_steps=100, sig=0.08, leray=False, w_mask=0.0)
    
    
    #utils.plot_report(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_shu, "super_ddpm_shu")
    #utils.plot_report(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_interp, "super_ddpm_interp")
    #utils.plot_report(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_mask, "super_ddpm_mask")
    #utils.plot_report(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_diff_mask, "super_ddpm_mask_diff")

    #utils.plot_report_3d(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_shu, "super_3d_ddpm_shu")
    #utils.plot_report_3d(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_interp, "super_3d_ddpm_interp")
    #utils.plot_report_3d(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_mask, "super_3d_ddpm_mask")
    #utils.plot_report_3d(dataset, config, [0, 4, 6, 7, 8], samples_x_norm, samples_y_norm, pred_diff_mask, "super_3d_ddpm_mask_diff")


