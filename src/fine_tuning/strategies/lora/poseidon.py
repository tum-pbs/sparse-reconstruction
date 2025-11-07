from GIFt.strategies.lora import LoRAConfigMixin, LoRALinearFineTuningStrategy, LoRAConvFineTuningStrategy
from GIFt.strategies import FineTuningStrategy, merger_strategy
from GIFt.modules.lora import LoRALinear
import GIFt.utils.factories as fts
        
class PoseidonLoRAAttentionStrategy(LoRAConfigMixin,FineTuningStrategy):
    
    def __init__(self, rank: int = 3, 
                 lora_alpha: float | None = None, 
                 lora_dropout: float = 0, 
                 train_bias: bool = False) -> None:
        LoRAConfigMixin.__init__(self,rank,lora_alpha,lora_dropout,train_bias)
        FineTuningStrategy.__init__(self,
                                    [
                                        (fts.mc_name_equal2_sequence(["query","key","value"]),
                                         fts.ma_replace(LoRALinear),
                                         self.lora_configs()),
                                    ],
                                    )