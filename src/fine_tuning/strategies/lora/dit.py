from GIFt.strategies import FineTuningStrategy,merger_strategy
from GIFt.strategies.lora import LoRALinearFineTuningStrategy,LoRAConvFineTuningStrategy
import GIFt.utils.factories as fts
from .generic import LoRAAllStrategy, LoRAFullStrategy
from typing import Sequence

class FullTrainLayers(FineTuningStrategy):
    def __init__(self,embedding_names: Sequence[str]) -> None:
        super().__init__()
        self.moule_caps.register_cap(
             fts.mc_cname_equal2_sequence(embedding_names),
            lambda *args,**kwargs: None,
        )
 
 
def DiTLoRAAllStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = False,
    full_train_embedding: bool = True,
    full_train_final_layer:bool = True) -> FineTuningStrategy:
    full_train_layers=[]
    if full_train_embedding:
        full_train_layers.append("pos_embed")
    if full_train_final_layer:
        full_train_layers.append("proj_out_2")
    if len(full_train_layers)>0:
        return merger_strategy([FullTrainLayers(full_train_layers),
                                LoRAAllStrategy(rank,lora_alpha,lora_dropout,train_bias)])
    else:
        return LoRAAllStrategy(rank,lora_alpha,lora_dropout,train_bias)
    
def DiTLoRAFullStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = True,
    full_train_embedding: bool = True,
    full_train_final_layer:bool = True) -> FineTuningStrategy:
    full_train_layers=[]
    if full_train_embedding:
        full_train_layers.append("pos_embed")
    if full_train_final_layer:
        full_train_layers.append("proj_out_2")
    if len(full_train_layers)>0:
        return merger_strategy([FullTrainLayers(full_train_layers),
                                LoRAFullStrategy(rank,lora_alpha,lora_dropout,train_bias)])
    else:
        return LoRAFullStrategy(rank,lora_alpha,lora_dropout,train_bias)