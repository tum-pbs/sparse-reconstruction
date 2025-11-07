import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import random
import importlib

import utils

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(1234)

def main():

    # Load the configuration
    print("Loading config...")
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    module_map = {
        "fm": "trainer_fm",
        "ddpm": "trainer_ddpm",
        "regression": "trainer_regression",
    }

    # Import and run the selected training module
    module_name = module_map[config.Type]
    train_module = importlib.import_module(module_name)
    
    print(f"Running {config.Type} training...")
    train_module.start_training()

if __name__ == "__main__":
    main()