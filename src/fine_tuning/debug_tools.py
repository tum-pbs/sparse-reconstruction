from main import get_config,get_parser
from omegaconf import OmegaConf
from src.setup import set_default_options
from src.utils import instantiate_from_config

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
def build_configs(config_path:str,return_dict=True):
    with suppress_stdout():
        parser=get_parser()
        parser.set_defaults(config=[config_path])
        opt, unknown = parser.parse_known_args()
        config = OmegaConf.create(get_config(opt, unknown))
        set_default_options(config)
        if return_dict:
            return OmegaConf.to_container(config)
        else:
            return config
    
def build_network(config_path=None,network_config=None):
    if network_config is not None:
        return instantiate_from_config(network_config)
    model_config=build_configs(config_path)["model"]
    return instantiate_from_config(model_config)