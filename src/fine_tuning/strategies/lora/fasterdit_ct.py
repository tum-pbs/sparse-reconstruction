from .generic import LoRAAllStrategy, LoRAFullStrategy
from GIFt.strategies import FineTuningStrategy
from GIFt.strategies.lora import LoRAConfigMixin
from GIFt.modules.lora import LoRALinear,LoRAConv2d
import GIFt.utils.factories as fts

def FasterDiTLoRAAllStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = False,) -> FineTuningStrategy:
    return LoRAAllStrategy(rank,lora_alpha,lora_dropout,train_bias)

def FasterDiTLoRAFullStrategy(
    rank: int = 3, 
    lora_alpha: float | None = None, 
    lora_dropout: float = 0, 
    train_bias: bool = True,) -> FineTuningStrategy:
    return LoRAFullStrategy(rank,lora_alpha,lora_dropout,train_bias)

class FasterDiTAttentionLoraStrategy(LoRAConfigMixin,FineTuningStrategy):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        LoRAConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self,
                                    [
                                        (fts.mc_name_equal2_sequence(["to_qkv"]),
                                         fts.ma_replace(LoRALinear),
                                         self.lora_configs()),
                                        (fts.mc_name_equal2_sequence(["to_out"]),
                                         fts.ma_replace(LoRAConv2d),
                                         self.lora_configs()),
                                    ]
                                    )