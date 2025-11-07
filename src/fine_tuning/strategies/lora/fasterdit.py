from .udit import UDiTLoRAAllStrategy,UDiTLoRAFullStrategy
from GIFt.strategies import FineTuningStrategy,merger_strategy
    
def FasterDiTLoRAAllStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = False,
    full_train_embedding: bool = True,
    full_train_final_layer:bool = True) -> FineTuningStrategy:
    return UDiTLoRAAllStrategy(rank,lora_alpha,lora_dropout,train_bias,full_train_embedding,full_train_final_layer)
    
def FasterDiTLoRAFullStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = True,
    full_train_embedding: bool = True,
    full_train_final_layer:bool = True) -> FineTuningStrategy:
    return UDiTLoRAFullStrategy(rank,lora_alpha,lora_dropout,train_bias,full_train_embedding,full_train_final_layer)