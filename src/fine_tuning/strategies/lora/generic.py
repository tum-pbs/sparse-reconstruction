from GIFt.strategies.lora import LoRALinearFineTuningStrategy, LoRAConvFineTuningStrategy
from GIFt.strategies import FineTuningStrategy, merger_strategy

def LoRAAllStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = False) -> FineTuningStrategy:
    return merger_strategy([LoRALinearFineTuningStrategy(rank,lora_alpha,lora_dropout,train_bias),
                            LoRAConvFineTuningStrategy(rank,lora_alpha,lora_dropout,train_bias)],)

def LoRAFullStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = True) -> FineTuningStrategy:
    return merger_strategy([LoRALinearFineTuningStrategy(rank,lora_alpha,lora_dropout,train_bias),
                            LoRAConvFineTuningStrategy(rank,lora_alpha,lora_dropout,train_bias)],
                            additional_para_caps=
                            [
                                (
                                lambda *args,**kwargs: True,
                                lambda *args,**kwargs: None,
                                {} 
                                )
                            ])