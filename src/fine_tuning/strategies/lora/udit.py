from GIFt.strategies import FineTuningStrategy,merger_strategy
import GIFt.utils.factories as fts
from .generic import LoRAAllStrategy, LoRAFullStrategy
from .dit import FullTrainLayers
    
def UDiTLoRAAllStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = False,
    full_train_embedding: bool = True,
    full_train_final_layer:bool = True) -> FineTuningStrategy:
    full_train_layers=[]
    if full_train_embedding:
        full_train_layers.append("x_embedder")
    if full_train_final_layer:
        full_train_layers.append("final_layer")
    if len(full_train_layers)>0:
        return merger_strategy([FullTrainLayers(full_train_layers),
                                LoRAAllStrategy(rank,lora_alpha,lora_dropout,train_bias)])
    else:
        return LoRAAllStrategy(rank,lora_alpha,lora_dropout,train_bias)
    
def UDiTLoRAFullStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = True,
    full_train_embedding: bool = True,
    full_train_final_layer:bool = True) -> FineTuningStrategy:
    full_train_layers=[]
    if full_train_embedding:
        full_train_layers.append("x_embedder")
    if full_train_final_layer:
        full_train_layers.append("final_layer")
    if len(full_train_layers)>0:
        return merger_strategy([FullTrainLayers(full_train_layers),
                                LoRAFullStrategy(rank,lora_alpha,lora_dropout,train_bias)])
    else:
        return LoRAFullStrategy(rank,lora_alpha,lora_dropout,train_bias)