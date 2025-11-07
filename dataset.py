import torch
import numpy as np
import utils 
from torch.utils.data import DataLoader, TensorDataset

class BigSpectralIsotropicTurbulenceDataset:
    def __init__(self, grid_size=128, norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10):
        self.grid_size = grid_size
        self.norm = norm
        self.size = size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        #self.data = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}.pt', weights_only=False, map_location='cpu')
        self.data = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_mindiv.pt', weights_only=False, map_location='cpu')
        #self.data = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_leray.pt', weights_only=False, map_location='cpu')
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = self.data.shape
        print(f"N_time: {self.N_time}, N_channels: {self.N_channels}, Nx: {self.Nx}, Ny: {self.Ny}, Nz: {self.Nz}")

        self.velocity = self.data[:, :3, :, :, :]
        self.data = self.velocity
        
        self.mean_velocity_field = self.velocity.mean(dim=0)
        
        mean_data, std_data = utils.compute_statistics(self.data)
        self.data_scaler = utils.StdScaler(mean_data, std_data)
        
        if self.norm:
            print("norm")
            self.data = self.data_scaler(self.data)
                
        if self.size is not None:
            self.N_time = self.size
            self.data = self.data[:self.size]
            
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(1234))
        #print(indices)
                
        train_size = int(self.train_ratio * self.N_time)
        val_size = int(self.val_ratio * self.N_time)
        test_size = self.N_time - train_size - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = torch.utils.data.Subset(self.data, train_indices)
        val_dataset = torch.utils.data.Subset(self.data, val_indices)
        test_dataset = torch.utils.data.Subset(self.data, test_indices)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        self.test_dataset = test_dataset[:self.num_samples]
    
    def __len__(self):
        return self.size
    
class SupervisedSpectralTurbulenceDataset:
    def __init__(self, grid_size=128, norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10):
        self.grid_size = grid_size
        self.norm = norm
        self.size = size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples

        # Load X (inputs) and Y (ground truth outputs)
        #self.X = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_mindiv_point1.pt', weights_only=False, map_location='cpu')
        self.X = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_mindiv_5.pt', weights_only=False, map_location='cpu')
        #self.X = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_mindiv_down4.pt', weights_only=False, map_location='cpu')
        #self.X = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_mindiv_15.pt', weights_only=False, map_location='cpu') 
        self.Y = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_mindiv.pt', weights_only=False, map_location='cpu')
        
        #self.X = torch.load("/mnt/data1/navarro/data_spectral_128_mindiv_combined.pt", weights_only=False, map_location='cpu')
        #self.Y = torch.load("/mnt/data1/navarro/data_spectral_128_mindiv_combined_gt.pt", weights_only=False, map_location='cpu')
        
        #self.X = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_5.pt', weights_only=False, map_location='cpu')
        #self.X = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}_down4.pt', weights_only=False, map_location='cpu') 
        #self.Y = torch.load(f'/mnt/data1/navarro/data_spectral_{grid_size}.pt', weights_only=False, map_location='cpu')

        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X)
        if isinstance(self.Y, np.ndarray):
            self.Y = torch.from_numpy(self.Y)
        
        self.Y = self.Y[:, :3, :, :, :]

        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = self.Y.shape
        print(f"N_time: {self.N_time}, N_channels: {self.N_channels}, Nx: {self.Nx}, Ny: {self.Ny}, Nz: {self.Nz}")

        mean_Y, std_Y = utils.compute_statistics(self.Y)
        self.Y_scaler = utils.StdScaler(mean_Y, std_Y)

        mean_X, std_X = utils.compute_statistics(self.X)
        self.X_scaler = utils.StdScaler(mean_X, std_X)

        if self.norm:
            self.X = self.X_scaler(self.X)
            self.Y = self.Y_scaler(self.Y)

        if self.size is not None:
            self.X = self.X[:self.size]
            self.Y = self.Y[:self.size]
            self.N_time = self.size
        else:
            self.size = self.N_time

        # Create consistent random split
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(1234))
        #print(indices)
        train_size = int(self.train_ratio * self.size)
        val_size = int(self.val_ratio * self.size)
        test_size = self.size - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create paired (X, Y) datasets
        train_dataset = TensorDataset(self.X[train_indices], self.Y[train_indices])
        val_dataset = TensorDataset(self.X[val_indices], self.Y[val_indices])
        test_dataset = TensorDataset(self.X[test_indices], self.Y[test_indices])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Sampled test set for visualization or evaluation
        self.test_dataset = test_dataset[:self.num_samples]

    def __len__(self):
        return self.size
    
    
