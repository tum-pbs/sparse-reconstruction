import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import os
import yaml
import numpy as np
import utils
import time

from dataset import SupervisedSpectralTurbulenceDataset
import utils
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def sparse_experiment(dataset, config, model, nsamples, samples_x, samples_y, leray=False):
    
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
    errors_y = []
    
    with torch.no_grad():
        for i in range(nsamples):
            print(f"Sample {i+1}/{nsamples}")
            x     = samples_x[i].unsqueeze(0).to(config.device)
            y     = samples_y[i].unsqueeze(0).to(config.device)

            start_time = time.time()
            y_pred = model(x)
            y_pred = y_pred.sample
            eval_time = time.time() - start_time
            print(f"Evaluation time: {eval_time:.4f} seconds")
            
            if dataset is not None:
                x = dataset.X_scaler.inverse(x)
                y = dataset.Y_scaler.inverse(y)
                y_pred = dataset.Y_scaler.inverse(y_pred)
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
            #                         f"super_regression_{i}")

            utils.experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, y_pred)
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"PSNR: {np.mean(psnr):.4f} +/- {np.std(psnr):.4f} (max: {np.max(psnr):.4f})")
    print(f"SSIM: {np.mean(ssim):.4f} +/- {np.std(ssim):.4f} (max: {np.max(ssim):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")

    return torch.stack(pred, dim=0).squeeze(1), torch.stack(samples_in, dim=0).squeeze(1), torch.stack(samples_out, dim=0).squeeze(1), errors_y

# Main script
if __name__ == "__main__":
    print("Loading config...")
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    # Generate samples using ODE integration
    num_samples = 10
    dataset = SupervisedSpectralTurbulenceDataset(grid_size=config.Data.grid_size,
                                                    norm=config.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=num_samples)
    
    samples_x, samples_y = dataset.test_dataset
    print(samples_y.shape)
    print(samples_x.shape)
    
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
    
    pred, samples_x, samples_y, rmse_profiles = sparse_experiment(dataset, config, model, num_samples, samples_x, samples_y, leray=False)
    #utils.plot_report(dataset, config, [0, 4, 6, 7, 8], samples_x, samples_y, pred, "super_regression_comb")
    #utils.plot_report_3d(dataset, config, [0, 4, 6, 7, 8], samples_x, samples_y, pred, "super_3d_regression_comb")