from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import os
from typing import List, Sequence

class WeightLoader(ABC):
    
    @abstractmethod
    def load(self, model:nn.Module, weight_path: str) -> nn.Module:
        pass
    
    
class PosiedonPreTrainedWeightLoader(WeightLoader):
    """
    Load pre-trained weights for Posiedon model.
    """
    
    def load(self, model:nn.Module, weight_path: str) -> nn.Module:
        assert os.path.isdir(weight_path), f"Weight path {weight_path} for Posiedon should be a directory."
        try:
            model.model.scot=model.model.scot.from_pretrained(weight_path,
                                                              config=model.model.scot.config,
                                                              ignore_mismatched_sizes=True)
            model.model.scot.train()
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weight_path} for Posiedon.") from e
        return model

class CKPTLoader(WeightLoader):
    """
    Load weights from a checkpoint file.
    """
    def __init__(self,strict: bool = True) -> None:
        super().__init__()
        self.strict = strict
    
    def load(self, model: nn.Module, weight_path: str) -> nn.Module:
        try:
            model.load_state_dict(torch.load(weight_path)['state_dict'], strict=self.strict)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weight_path} for model.") from e
        return model
    
class StatDictLoader(WeightLoader):
    """
    Load weights from a state_dict file.
    """
    def __init__(self,strict: bool = True) -> None:
        super().__init__()
        self.strict = strict
    
    def load(self, model: nn.Module, weight_path: str) -> nn.Module:
        try:
            model.load_state_dict(torch.load(weight_path), strict=self.strict)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weight_path} for model.") from e
        return model
    
class _LayerIgnoreLoader(WeightLoader):
    """
    Ignore loading weights for embedding layer.
    """
    def __init__(self, ignored_layers: Sequence[str]) -> None:
        super().__init__()
        self.ignore_layers = ignored_layers
    
    def load(self, model: nn.Module, weight_path: str) -> nn.Module:
        trained_weights = torch.load(weight_path)['state_dict']
        network_state_dict = model.state_dict()
        for key in network_state_dict.keys():
            if not self._contains(key, self.ignore_layers):
                network_state_dict[key] = trained_weights[key]
        model.load_state_dict(network_state_dict)
        return model
    
    def _contains(self,key:str, ignore_names: list) -> bool:
        for name in ignore_names:
            if name in key:
                return True
        return False
    
def DiTWeightLoader(load_embedding: bool = False,
                    load_final_layer:bool = False) -> WeightLoader:
    """
    Load weights for DIT model.
    """
    ignored_layers = []
    if not load_embedding:
        ignored_layers.append("pos_embed")
    if not load_final_layer:
        ignored_layers.append("proj_out_2")
    if len(ignored_layers) == 0:
        return CKPTLoader()
    else:
        return _LayerIgnoreLoader(ignored_layers)
    
def UDiTWeightLoader(load_embedding: bool = False,
                    load_final_layer:bool = False) -> WeightLoader:
    """
    Load weights for U-DIT model.
    """
    ignored_layers = []
    if not load_embedding:
        ignored_layers.append("x_embedder")
    if not load_final_layer:
        ignored_layers.append("final_layer")
    if len(ignored_layers) == 0:
        return CKPTLoader()
    else:
        return _LayerIgnoreLoader(ignored_layers)
    
def FasterDiTV1WeightLoader(load_embedding: bool = False,
                    load_final_layer:bool = False) -> WeightLoader:
    """
    Load weights for Faster-DIT model.
    """
    return UDiTWeightLoader(load_embedding, load_final_layer)
    