import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import os
import yaml
import numpy as np
import utils

from dataset import SupervisedSpectralTurbulenceDataset
import utils

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def experiment(dataset, config, nsamples, samples_x, samples_y):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    spectrum = []
    psnr = []
    ssim = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)

        utils.experiment_sr(dataset, config, i, losses, psnr, ssim, residuals, residuals_gt, residuals_diff, lsim, blurriness, spectrum, y, x)
        #utils.plot_slice(x.cpu(), 0, 0, int(128/2), name=f"interp_{i}")
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"PSNR: {np.mean(psnr):.4f} +/- {np.std(psnr):.4f} (max: {np.max(psnr):.4f})")
    print(f"SSIM: {np.mean(ssim):.4f} +/- {np.std(ssim):.4f} (max: {np.max(ssim):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4f} +/- {np.std(spectrum):.4f} (max: {np.max(spectrum):.4f})")

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
    
    samples_x, ids = utils.interpolate_dataset(samples_y, 5/100, method="nearest") # 5% sparse data
    #samples_x, ids = utils.downscale_data(samples_y, 4, order=0) # Super-resolution 4x
    
    if dataset is not None:
        samples_x = dataset.X_scaler.inverse(samples_x)
        samples_y = dataset.X_scaler.inverse(samples_y)
    
    experiment(dataset, config, num_samples, samples_x, samples_y)